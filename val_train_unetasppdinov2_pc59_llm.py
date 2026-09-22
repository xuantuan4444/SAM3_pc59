#!/usr/bin/env python3
"""
val_train_unetasppdinov2_pc59_llm.py
=======================================
Standalone script version of val_train_unetasppdinov2_pc59_llm.ipynb.

"Hybrid" evaluation: SAM3 runs the full 59-class evaluation (LLM prompt ensemble, from
adjust_prompt_pc59.json, for whichever classes have an entry -- every other class falls back to
its own bare name), while the trained SAM3+UNet+ASPP+DINOv2 model hybrid-refines only the N
target classes it was trained on (read dynamically from the checkpoint, NOT hardcoded here,
DINOv2 config included). At every pixel where the refine network predicts something other than
"other", the result overwrites the SAM3 baseline; everywhere else, the SAM3 baseline is kept
unchanged.

Unlike train_unetasppdinov2_pc59_llm.py, DINOv2 features are computed ON-THE-FLY per image (no
cache) -- val images are each visited once, so caching would add complexity for no benefit.

PC59 uses a 0-58/255-void label convention (no valid "background" class in the label space):
sam3_pred defaults to 255 (no confident detection), NOT 0, since 0 is a real class ("aeroplane").

Logic is unchanged from the notebook -- only I/O has been adapted for a plain VM with no Jupyter
(plt.show()/display() -> saved PNG/CSV files, no notebook magics).

--------------------------------------------------------------------------------------------
Expected project layout (all paths resolved from this script's own location):

    pc59_ablation/                              <- PROJECT_ROOT
    |-- val_train_unetasppdinov2_pc59_llm.py    <- this file
    |-- data_pc59/
    |   |-- JPEGImages/
    |   |-- SegmentationClassContext/
    |   |-- pascal_context_val.txt
    |   `-- adjust_prompt_pc59.json             <- optional; falls back to bare names if missing
    |-- weight_sam3/sam3.pt
    `-- weights_aspp_pc59_llm_v1/unet_aspp_pc59_llm_v1_best.pth   <- output of train_unetasppdinov2_pc59_llm.py

DINOv2 weights: downloaded automatically via torch.hub on first run (needs internet), or loaded
from a local repo/checkpoint if torch.hub.load fails (see load_dinov2_backbone() below).

Output (written under this script's folder):
    per_class_iou_bar_chart.png
    class_visualizations/<class_name>.png
    summary_metrics.csv
    per_class_metrics.csv

Setup (run once on the VM, before this script -- NOT executed by this script itself):

    pip install torch torchvision
    pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
    pip install iopath ftfy portalocker pandas matplotlib pillow numpy tqdm opencv-python-headless

Usage:

    python val_train_unetasppdinov2_pc59_llm.py

If your data/checkpoint don't match the layout above, edit the CONFIG section below directly.
--------------------------------------------------------------------------------------------
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import time
import traceback
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as tvm
from torchvision.models import ResNet34_Weights
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# CONFIG
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PC59_IMAGES_ROOT = PROJECT_ROOT / "data_pc59" / "JPEGImages"
PC59_GT_ROOT = PROJECT_ROOT / "data_pc59" / "SegmentationClassContext"
VAL_FILE = PROJECT_ROOT / "data_pc59" / "pascal_context_val.txt"
ADJUST_PROMPT_PATH = PROJECT_ROOT / "data_pc59" / "adjust_prompt_pc59.json"
SAM3_CKPT = PROJECT_ROOT / "weight_sam3" / "sam3.pt"
UNET_CKPT_PATH = PROJECT_ROOT / "weights_aspp_pc59_llm_v1" / "unet_aspp_pc59_llm_v1_best.pth"  # TODO: adjust -- output of train_unetasppdinov2_pc59_llm.py

OUT_BAR_CHART = PROJECT_ROOT / "per_class_iou_bar_chart.png"
OUT_VIS_DIR = PROJECT_ROOT / "class_visualizations"
OUT_SUMMARY_CSV = PROJECT_ROOT / "summary_metrics.csv"
OUT_CLASS_CSV = PROJECT_ROOT / "per_class_metrics.csv"

VOID_VALUE = 255  # canonical ignore value (see sam3_baseline_pc59_llm.py) -- PC59's raw masks
                   # have no genuine void; 0 there means "background/other", which becomes 255
                   # (ignore, never a valid prediction target) after normalize_gt() below.

COARSE_THRESHOLDS = [0.50, 0.30, 0.20, 0.15]
CONFIDENCE_THRESHOLD = 0.30
MAX_VAL_IMAGES = None  # None = all val images found; set an int for a quick smoke-test run
NUM_SAMPLE_VIS = 2      # up to N example images stored per class for visualization

# 59 classes, 0-indexed (0..58) -- verified against mmsegmentation's PascalContextDataset59.CLASSES,
# same list used throughout this ablation.
PC59_CLASSES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book",
    "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth",
    "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food",
    "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse",
    "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves",
    "sidewalk", "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood",
]
assert len(PC59_CLASSES) == 59
NUM_CLASSES = len(PC59_CLASSES)  # 59 -- no background slot in this label space
INDEX_TO_CLASS = {i: name for i, name in enumerate(PC59_CLASSES)}
INDEX_TO_CLASS_INV = {name: idx for idx, name in INDEX_TO_CLASS.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Populated by build_sam3() in main()
DEVICE = None
processor = None

# Populated by load_prompts_and_val_ids() in main()
CLASS_PROMPTS_OVERRIDE = None
val_ids = None

# Populated by load_unet_checkpoint() in main()
unet_model = None
TARGET_CLASSES = None
TARGET_TO_LOCAL = None
LOCAL_TO_TARGET = None
LOCAL_TO_GLOBAL_IDX = None
NUM_TARGETS = None
NUM_OUT_CLASSES = None
IN_CHANNELS_BASE = None
IMAGE_H = None
IMAGE_W = None
DINOV2_MODEL_NAME = None
DINOV2_EMBED_DIM = None
DINOV2_INPUT_SIZE = None
DINOV2_GRID_SIZE = None
DINOV2_COMPRESS = None

# Populated by load_dinov2_backbone() in main()
dinov2_model = None


def normalize_gt(raw_mask):
    """Raw PC59 mask (0=background/other, 1-59=class) -> canonical form (0-58=class, 255=ignore)."""
    out = raw_mask.astype(np.int16) - 1
    out[out < 0] = VOID_VALUE
    return out.astype(np.uint8)


def load_pc59_mask(mask_path):
    """Raw GT mask -> uint8 label map, 0=background/other, 1-59=class (see normalize_gt above)."""
    return np.array(Image.open(mask_path), dtype=np.uint8)


def extract_masks_from_state(state):
    """Read masks & scores from the SAM3 state dict after set_text_prompt()."""
    if "masks" not in state or "scores" not in state:
        return [], []
    masks_tensor = state["masks"]
    scores_tensor = state["scores"]
    if masks_tensor is None or len(masks_tensor) == 0:
        return [], []
    out_masks, out_scores = [], []
    for i in range(len(masks_tensor)):
        mask = masks_tensor[i].squeeze(0).cpu().numpy()
        score = scores_tensor[i].item()
        out_masks.append(mask)
        out_scores.append(score)
    return out_masks, out_scores


def union_at_threshold(masks, scores, shape_hw, thresholds):
    """Union-at-threshold -- same as get_coarse_pc59_llm.py: try each threshold in order, use
    the first one with >=1 mask, union all masks reaching that threshold."""
    h, w = shape_hw
    scores_np = np.asarray(scores, dtype=np.float32)
    for thr in thresholds:
        idx = np.where(scores_np >= thr)[0]
        if len(idx) > 0:
            union = np.zeros((h, w), dtype=bool)
            for i in idx:
                union |= masks[i]
            return union.astype(np.uint8), thr
    return np.zeros((h, w), dtype=np.uint8), None


def fast_confusion_matrix(gt, pred, num_classes):
    """Confusion matrix, ignoring VOID_VALUE."""
    valid = (gt != VOID_VALUE) & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes * num_classes,
    )
    return hist.reshape(num_classes, num_classes)


# ============================================================================
# Setup
# ============================================================================

def setup_torch():
    global DEVICE
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)
    print("Imports OK, global autocast bfloat16 enabled")


def build_sam3():
    global processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    sam3_dir = Path(sam3.__file__).parent
    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    sam3_model = build_sam3_image_model(bpe_path=str(bpe_path), checkpoint_path=str(SAM3_CKPT), load_from_HF=False)
    sam3_model = sam3_model.to(DEVICE)  # FP32 -- autocast handles bfloat16 per-op

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = Sam3Processor(sam3_model, confidence_threshold=min(COARSE_THRESHOLDS))
    print("SAM3 model loaded.")
    print(f"Processor confidence_threshold: {min(COARSE_THRESHOLDS)} (low, to allow coarse fallback)")
    print(f"Owned-class competition threshold (applied in code): {CONFIDENCE_THRESHOLD}")


def load_prompts_and_val_ids():
    global CLASS_PROMPTS_OVERRIDE, val_ids

    if ADJUST_PROMPT_PATH.exists():
        with open(ADJUST_PROMPT_PATH, encoding="utf-8") as f:
            adjust_prompt = json.load(f)
    else:
        print(f"[warn] {ADJUST_PROMPT_PATH} not found -- every class falls back to its own name as the single prompt.")
        adjust_prompt = {}

    CLASS_PROMPTS_OVERRIDE = {
        name: prompts
        for name, prompts in adjust_prompt.items()
        if name in PC59_CLASSES and len(prompts) > 0
    }
    print("Prompt ensembles (only overridden classes shown):")
    for name, prompts in CLASS_PROMPTS_OVERRIDE.items():
        print(f"  {name}: {prompts}")

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        # "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png" per line
        val_ids = [Path(line.strip().split()[0]).stem for line in f if line.strip()]

    print(f"PC59 images: {PC59_IMAGES_ROOT}")
    print(f"Val images: {len(val_ids)}")
    print(f"Classes ({len(PC59_CLASSES)}): {PC59_CLASSES}")


# ============================================================================
# Model: ResNet-34 + ASPP + UNet decoder (no DINOv2) -- must match
# train_unetaspp_pc59_llm.py exactly for load_state_dict to succeed.
# ============================================================================

class ASPP(nn.Module):
    def __init__(self, in_ch=512, out_ch=256):
        super().__init__()
        def _branch(dilation):
            if dilation == 1:
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b1 = _branch(1)
        self.b6 = _branch(6)
        self.b12 = _branch(12)
        self.b18 = _branch(18)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        gap = F.interpolate(self.gap(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([self.b1(x), self.b6(x), self.b12(x), self.b18(x), gap], dim=1))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ResNet34UNetASPPPC59(nn.Module):
    def __init__(self, in_channels_base=9, dinov2_dim=384, dinov2_compress=32, out_channels=7):
        super().__init__()
        self.total_in = in_channels_base + dinov2_compress
        self.dino_compress = nn.Sequential(
            nn.Conv2d(dinov2_dim, dinov2_compress, 1, bias=False),
            nn.BatchNorm2d(dinov2_compress),
            nn.ReLU(inplace=True),
        )
        backbone = tvm.resnet34(weights=None)  # weights loaded from checkpoint, no ImageNet needed
        new_conv = nn.Conv2d(self.total_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1 = new_conv
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4
        self.aspp = ASPP(in_ch=512, out_ch=256)
        self.dec4 = DecoderBlock(256, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, input_base, dino_feat):
        H, W = input_base.shape[-2:]
        dino_c = self.dino_compress(dino_feat)
        dino_u = F.interpolate(dino_c, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([input_base, dino_u], dim=1)
        e0 = self.enc0(x)
        ep = self.pool(e0)
        e1 = self.enc1(ep)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bottle = self.aspp(e4)
        d = self.dec4(bottle, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d, e0)
        d = self.final_up(d)
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
        return self.head(d)


def load_unet_checkpoint():
    global unet_model, TARGET_CLASSES, TARGET_TO_LOCAL, LOCAL_TO_TARGET, LOCAL_TO_GLOBAL_IDX
    global NUM_TARGETS, NUM_OUT_CLASSES, IN_CHANNELS_BASE, IMAGE_H, IMAGE_W
    global DINOV2_MODEL_NAME, DINOV2_EMBED_DIM, DINOV2_INPUT_SIZE, DINOV2_GRID_SIZE, DINOV2_COMPRESS

    unet_ckpt = torch.load(UNET_CKPT_PATH, map_location=DEVICE)

    # Read the full config back from the checkpoint -- NOT hardcoded, to avoid drifting from train time
    TARGET_CLASSES = unet_ckpt["target_classes"]
    TARGET_TO_LOCAL = unet_ckpt["target_to_local"]
    LOCAL_TO_TARGET = {v: k for k, v in TARGET_TO_LOCAL.items()}
    NUM_TARGETS = len(TARGET_CLASSES)
    NUM_OUT_CLASSES = unet_ckpt["num_out_classes"]
    IN_CHANNELS_BASE = unet_ckpt["in_channels_base"]
    DINOV2_MODEL_NAME = unet_ckpt["dinov2_model_name"]
    DINOV2_EMBED_DIM = unet_ckpt["dinov2_embed_dim"]
    DINOV2_INPUT_SIZE = unet_ckpt["dinov2_input_size"]
    DINOV2_GRID_SIZE = unet_ckpt["dinov2_grid_size"]
    DINOV2_COMPRESS = unet_ckpt["dinov2_compress"]
    IMAGE_H = unet_ckpt["image_h"]
    IMAGE_W = unet_ckpt["image_w"]

    unet_model = ResNet34UNetASPPPC59(
        in_channels_base=IN_CHANNELS_BASE,
        dinov2_dim=DINOV2_EMBED_DIM,
        dinov2_compress=DINOV2_COMPRESS,
        out_channels=NUM_OUT_CLASSES,
    ).to(DEVICE)
    unet_model.load_state_dict(unet_ckpt["model_state_dict"])
    unet_model.eval()
    for p in unet_model.parameters():
        p.requires_grad = False

    print(f"Loaded UNet+ASPP+DINOv2 checkpoint: {UNET_CKPT_PATH}")
    print(f"  arch_version   : {unet_ckpt.get('arch_version')}")
    print(f"  trained epoch  : {unet_ckpt.get('epoch')}   internal-val mIoU: {unet_ckpt.get('miou'):.4f}")
    print(f"  TARGET_CLASSES ({NUM_TARGETS}): {TARGET_CLASSES}")
    print(f"  Image size (train): {IMAGE_H} x {IMAGE_W}")
    print(f"  IN_CHANNELS_BASE   : {IN_CHANNELS_BASE}   DINOv2: {DINOV2_MODEL_NAME} ({DINOV2_EMBED_DIM}d, grid {DINOV2_GRID_SIZE}x{DINOV2_GRID_SIZE})")

    # SAM3 is NOT limited to just N classes -- it runs the full 59-class evaluation.
    # TARGET_CLASSES (read from the checkpoint) is only used to: (1) know which classes
    # UNet+ASPP will refine, (2) map local id 1..N -> global index 0..58, (3) label "Source".
    LOCAL_TO_GLOBAL_IDX = {local_id: INDEX_TO_CLASS_INV[name] for local_id, name in LOCAL_TO_TARGET.items()}
    print(f"UNet+ASPP hybrid-refine for {len(TARGET_CLASSES)} class(es): {TARGET_CLASSES}")
    print(f"The remaining {59 - len(TARGET_CLASSES)} classes stay entirely SAM3 baseline (unchanged):")
    print(f"  {[c for c in PC59_CLASSES if c not in TARGET_CLASSES]}")


def load_dinov2_backbone():
    """Load DINOv2 backbone (frozen) -- val images run once each, so compute on-the-fly, no
    cache (unlike train_unetasppdinov2_pc59_llm.py, which caches since each image is visited
    across many epochs)."""
    global dinov2_model
    print(f"Loading DINOv2 ({DINOV2_MODEL_NAME})...")
    try:
        dinov2_model = torch.hub.load("facebookresearch/dinov2", DINOV2_MODEL_NAME, trust_repo=True)
    except Exception as e:
        print(f"torch.hub.load failed: {e}")
        print("Fallback: trying to load from a local repo/checkpoint...")
        import sys as _sys
        dinov2_repo_candidates = [
            PROJECT_ROOT / "dinov2-repo",
            Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
        ]
        repo_path = next((p for p in dinov2_repo_candidates if p.exists()), None)
        if repo_path is None:
            raise RuntimeError(f"DINOv2 repo not found. Tried: {dinov2_repo_candidates}.")
        _sys.path.insert(0, str(repo_path))
        from dinov2.hub.backbones import dinov2_vits14 as _dinov2_build
        dinov2_model = _dinov2_build(pretrained=False)
        w_candidates = [
            PROJECT_ROOT / "weight_sam3" / "dinov2_vits14_pretrain.pth",
            repo_path / "dinov2_vits14_pretrain.pth",
        ]
        wpath = next((p for p in w_candidates if p.exists()), None)
        if wpath is None:
            raise RuntimeError(f"DINOv2 weights not found. Tried: {w_candidates}")
        sd = torch.load(wpath, map_location="cpu")
        dinov2_model.load_state_dict(sd)
        print(f"Loaded DINOv2 weights from {wpath}")

    dinov2_model = dinov2_model.to(DEVICE).eval()
    for p in dinov2_model.parameters():
        p.requires_grad = False

    print(f"DINOv2 {DINOV2_MODEL_NAME} loaded (frozen)")


@torch.no_grad()
def dinov2_forward(image_pil):
    """PIL image -> DINOv2 patch tokens -> feature map [1, DINOV2_EMBED_DIM, grid, grid]
    float32 GPU tensor."""
    img_r = image_pil.convert("RGB").resize((DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), Image.BILINEAR)
    x = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(DEVICE)
    out = dinov2_model.forward_features(x)
    tokens = out["x_norm_patchtokens"]
    B, N, D = tokens.shape
    feat = tokens.transpose(1, 2).reshape(B, D, DINOV2_GRID_SIZE, DINOV2_GRID_SIZE)
    return feat


# ============================================================================
# Hybrid predict
# ============================================================================

@torch.no_grad()
def predict_hybrid_semantic_map(image_path):
    """
    Returns final_pred (H, W) uint8:
      0-58 = global PC59 class index (INDEX_TO_CLASS)
      255  = no confident detection from ANY class (never a valid prediction target -- unlike
             VOC/Cityscapes, PC59 has no "background" class for this to fall back to)

    SAM3 branch: cross-class argmax over the FULL 59 classes, not limited in any way -- the
    unchanged baseline.

    UNet+ASPP+DINOv2 branch: only active on the N classes in TARGET_CLASSES. The result
    OVERWRITES the SAM3 baseline at every pixel where it predicts something other than "other".
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    state = processor.set_image(image)  # encode backbone once, shared by both branches

    # -- SAM3 baseline branch: FULL 59 classes, global cross-class argmax, threshold 0.3 --------
    sam3_pred = np.full((height, width), VOID_VALUE, dtype=np.uint8)
    best_score = np.zeros((height, width), dtype=np.float32)

    coarse_dict = {}

    for class_idx, cname in enumerate(PC59_CLASSES):  # 0-indexed, no offset
        prompts = CLASS_PROMPTS_OVERRIDE.get(cname, [cname])
        is_target = cname in TARGET_CLASSES

        class_score = np.zeros((height, width), dtype=np.float32)
        class_hit = np.zeros((height, width), dtype=bool)
        all_masks_raw, all_scores_raw = ([], []) if is_target else (None, None)

        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = extract_masks_from_state(state)

            if is_target:
                all_masks_raw.extend(masks)
                all_scores_raw.extend(scores)

            for mask, score in zip(masks, scores):
                if score < CONFIDENCE_THRESHOLD:
                    continue
                update = mask & (score > class_score)
                class_score[update] = score
                class_hit[update] = True

        update = class_hit & (class_score > best_score)
        sam3_pred[update] = class_idx
        best_score[update] = class_score[update]

        if is_target:
            union, _ = union_at_threshold(all_masks_raw, all_scores_raw, (height, width), COARSE_THRESHOLDS)
            coarse_dict[cname] = union

    # -- UNet+ASPP branch: forward on the N-channel coarse just gathered ------------------------
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
    coarse_stack = np.stack([
        cv2.resize(coarse_dict[c], (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
        for c in TARGET_CLASSES
    ], axis=0).astype(np.float32)

    rgb_t = TF.normalize(TF.to_tensor(image_resized), IMAGENET_MEAN, IMAGENET_STD)
    coarse_t = torch.from_numpy(coarse_stack)
    input_base = torch.cat([rgb_t, coarse_t], dim=0).unsqueeze(0).to(DEVICE)

    dino_feat = dinov2_forward(image)  # [1, DINOV2_EMBED_DIM, grid, grid] on DEVICE

    logits = unet_model(input_base, dino_feat)
    local_pred_small = logits.argmax(dim=1).squeeze(0).byte().cpu().numpy()
    local_pred = cv2.resize(local_pred_small, (width, height), interpolation=cv2.INTER_NEAREST)

    # -- Merge (hybrid) --------------------------------------------------------------------------
    final_pred = sam3_pred.copy()
    for local_id, cname in enumerate(TARGET_CLASSES, start=1):
        global_idx = LOCAL_TO_GLOBAL_IDX[local_id]
        sel = local_pred == local_id
        final_pred[sel] = global_idx

    return final_pred


# ============================================================================
# Evaluation loop
# ============================================================================

def run_evaluation():
    run_ids = val_ids if MAX_VAL_IMAGES is None else val_ids[:MAX_VAL_IMAGES]
    total = len(run_ids)

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    errors = []

    start_time = time.time()
    pbar = tqdm(run_ids, desc="Evaluating PC59 val (hybrid)", dynamic_ncols=True)
    for idx, image_id in enumerate(pbar):
        image_path = PC59_IMAGES_ROOT / f"{image_id}.jpg"
        mask_path = PC59_GT_ROOT / f"{image_id}.png"

        if not image_path.exists() or not mask_path.exists():
            tqdm.write(f"[{idx+1:4d}/{total}] SKIP {image_id} -- file not found")
            continue

        t0 = time.time()
        try:
            gt = normalize_gt(load_pc59_mask(mask_path))
            pred = predict_hybrid_semantic_map(image_path)
            t_img = time.time() - t0

            conf_mat_local = fast_confusion_matrix(gt, pred, num_classes=NUM_CLASSES)
            conf_mat += conf_mat_local

            gt_classes = set(np.unique(gt)) - {VOID_VALUE}
            gt_classes = {c for c in gt_classes if 0 <= c < NUM_CLASSES}
            for cls_idx in gt_classes:
                if len(class_samples[cls_idx]) < NUM_SAMPLE_VIS:
                    class_samples[cls_idx].append({
                        "id": image_id, "image_path": str(image_path), "gt": gt, "pred": pred,
                    })

            cls_names = [INDEX_TO_CLASS[c] for c in sorted(gt_classes)]
            pbar.set_postfix({"img": image_id, "t": f"{t_img:.1f}s"})
            tqdm.write(f"[{idx+1:4d}/{total}] {image_id}  |  classes: {cls_names}  |  {t_img:.1f}s")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception:
            t_img = time.time() - t0
            errors.append((image_id, "see traceback above"))
            tqdm.write(f"[{idx+1:4d}/{total}] ERROR {image_id} ({t_img:.1f}s):")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done. {total} images in {elapsed / 60:.2f} min ({elapsed / max(total, 1):.2f} s/img)")
    if errors:
        print(f"{len(errors)} images failed:")
        for img_id, err in errors[:5]:
            print(f"  {img_id}: {err}")

    return conf_mat, class_samples


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(conf_mat):
    eps = 1e-10
    diag = np.diag(conf_mat).astype(np.float64)
    gt_sum = conf_mat.sum(axis=1).astype(np.float64)
    pred_sum = conf_mat.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag

    iou = diag / (union + eps)
    dice = (2.0 * diag) / (gt_sum + pred_sum + eps)
    pixel_acc = diag.sum() / (conf_mat.sum() + eps)

    miou = np.nanmean(iou)
    mdice = np.nanmean(dice)

    summary_df = pd.DataFrame({
        "Metric": ["Pixel Accuracy", "mIoU (59 classes)", "Mean Dice (59 classes)"],
        "Value": [pixel_acc, miou, mdice],
    })

    class_df = pd.DataFrame({
        "Class": [INDEX_TO_CLASS[i] for i in range(NUM_CLASSES)],
        "Source": ["SAM3 + UNet+ASPP (hybrid)" if INDEX_TO_CLASS[i] in TARGET_CLASSES else "SAM3 (baseline)"
                   for i in range(NUM_CLASSES)],
        "IoU": iou[0:NUM_CLASSES],
        "Dice": dice[0:NUM_CLASSES],
        "GT Pixels": gt_sum[0:NUM_CLASSES].astype(np.int64),
        "Pred Pixels": pred_sum[0:NUM_CLASSES].astype(np.int64),
    }).sort_values("IoU", ascending=True).reset_index(drop=True)

    print("=" * 50)
    print(f"SUMMARY METRICS (SAM3 full 59-class baseline + UNet+ASPP hybrid-refine on {NUM_TARGETS} class(es), no DINOv2)")
    print("=" * 50)
    print(summary_df.to_string(index=False, formatters={"Value": lambda x: f"{x:.4f}"}))

    print(f"\n{'='*50}\nPER-CLASS METRICS (sorted by IoU ascending)\n{'='*50}")
    print(class_df.to_string(index=False, formatters={"IoU": lambda x: f"{x:.4f}", "Dice": lambda x: f"{x:.4f}"}))

    print(f">>> mIoU hybrid: {miou:.4f}")

    print("\n" + "=" * 50 + "\n5 WEAKEST CLASSES (lowest IoU)\n" + "=" * 50)
    print(class_df.head(5).to_string(index=False))
    print("\n" + "=" * 50 + "\n5 STRONGEST CLASSES (highest IoU)\n" + "=" * 50)
    print(class_df.tail(5).to_string(index=False))

    hybrid_rows = class_df[class_df["Source"] == "SAM3 + UNet+ASPP (hybrid)"]
    baseline_rows = class_df[class_df["Source"] == "SAM3 (baseline)"]

    print(f"\n>>> mIoU (all 59 classes)                          : {miou:.4f}")
    print(f">>> Pixel Accuracy                                  : {pixel_acc:.4f}")
    print(f">>> Mean Dice (all 59 classes)                       : {mdice:.4f}")
    print(f">>> mIoU trung binh {NUM_TARGETS} class (SAM3+UNet+ASPP hybrid)      : {hybrid_rows['IoU'].mean():.4f}")
    print(f">>> mIoU trung binh {59 - NUM_TARGETS} class con lai (SAM3 baseline) : {baseline_rows['IoU'].mean():.4f}")

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    class_df.to_csv(OUT_CLASS_CSV, index=False)
    print(f"\nSaved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_CLASS_CSV}")

    return iou, class_df, miou


# ============================================================================
# Plots
# ============================================================================

def plot_bar_chart(class_df, miou):
    fig, ax = plt.subplots(figsize=(10, max(10, NUM_CLASSES * 0.2)))
    colors = plt.cm.RdYlGn(class_df["IoU"].values)
    bars = ax.barh(class_df["Class"], class_df["IoU"], color=colors)

    for bar, src in zip(bars, class_df["Source"].values):
        if src == "SAM3 + UNet+ASPP (hybrid)":
            bar.set_edgecolor("black")
            bar.set_linewidth(1.8)
            bar.set_hatch("//")

    ax.set_xlabel("IoU", fontsize=12)
    ax.set_title(f"PC59 Val -- Per-class IoU (SAM3 baseline + UNet+ASPP hybrid-refine, mIoU={miou:.4f})", fontsize=13)
    ax.set_xlim(0, 1)
    ax.axvline(x=miou, color="blue", linestyle="--", linewidth=1, label=f"mIoU = {miou:.4f}")

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label=f"SAM3 + UNet+ASPP hybrid-refine ({NUM_TARGETS} class trained)"),
        Patch(facecolor="white", edgecolor="gray", label="SAM3 baseline, unchanged (remaining classes)"),
        plt.Line2D([0], [0], color="blue", linestyle="--", label=f"mIoU = {miou:.4f}"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    for bar, val in zip(bars, class_df["IoU"].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT_BAR_CHART, dpi=120)
    plt.close(fig)
    print(f"Saved: {OUT_BAR_CHART}")


def get_generic_colormap(n):
    """Deterministic VOC-style bit-interleaved palette for n classes."""
    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


def colorize_mask(mask, cmap):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in range(cmap.shape[0]):
        colored[mask == label_id] = cmap[label_id]
    colored[mask == VOID_VALUE] = [128, 128, 128]
    return colored


def plot_class_visualizations(class_samples, iou):
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    pc59_cmap = get_generic_colormap(NUM_CLASSES)

    for cls_idx in range(NUM_CLASSES):
        cls_name = INDEX_TO_CLASS[cls_idx]
        samples = class_samples.get(cls_idx, [])
        if not samples:
            continue

        n_show = min(2, len(samples))
        fig, axes = plt.subplots(n_show, 4, figsize=(18, 4.5 * n_show))
        if n_show == 1:
            axes = axes[np.newaxis, :]

        src_tag = "SAM3 + UNet+ASPP hybrid" if cls_name in TARGET_CLASSES else "SAM3 baseline"
        fig.suptitle(f"Class: {cls_name} (IoU = {iou[cls_idx]:.4f})  [{src_tag}]", fontsize=14, fontweight="bold")

        for row in range(n_show):
            s = samples[row]
            img = np.array(Image.open(s["image_path"]).convert("RGB"))
            gt_c = colorize_mask(s["gt"], pc59_cmap)
            pred_c = colorize_mask(s["pred"], pc59_cmap)
            err = (s["pred"] != s["gt"]) & (s["gt"] != VOID_VALUE)

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(f"Image: {s['id']}", fontsize=10)
            axes[row, 0].axis("off")

            axes[row, 1].imshow(gt_c)
            axes[row, 1].set_title("Ground Truth", fontsize=10)
            axes[row, 1].axis("off")

            axes[row, 2].imshow(pred_c)
            axes[row, 2].set_title("Prediction (hybrid)", fontsize=10)
            axes[row, 2].axis("off")

            axes[row, 3].imshow(img)
            axes[row, 3].imshow(err.astype(np.float32), alpha=0.55, cmap="Reds")
            axes[row, 3].set_title("Error Overlay", fontsize=10)
            axes[row, 3].axis("off")

        plt.tight_layout()
        out_path = OUT_VIS_DIR / f"{cls_name}.png"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)

    print(f"Saved per-class visualizations to {OUT_VIS_DIR}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    setup_torch()
    build_sam3()
    load_prompts_and_val_ids()
    load_unet_checkpoint()
    load_dinov2_backbone()

    conf_mat, class_samples = run_evaluation()
    iou, class_df, miou = compute_metrics(conf_mat)
    plot_bar_chart(class_df, miou)
    plot_class_visualizations(class_samples, iou)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] val_train_unetasppdinov2_pc59_llm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
