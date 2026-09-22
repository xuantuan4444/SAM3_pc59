#!/usr/bin/env python3
"""
train_unetaspp_pc59_nollm.py
==============================
Standalone script version of train_unetaspp_pc59_nollm.ipynb.

Trains the SAM3+UNet+ASPP refinement network (no DINOv2) on PASCAL-Context 59, no-LLM (bare
class-name prompt) coarse masks. Logic is unchanged from the notebook -- only I/O has been
adapted for a plain VM with no Jupyter (plt.show() -> saved PNG, no notebook magics).

TARGET_CLASSES is still sourced from adjust_prompt_pc59.json's keys (the SAME file the LLM-arm
script reads) -- unlike VOC/Cityscapes' no-LLM scripts, which hardcode a literal target-class
list, PC59's target classes are not yet decided, so both arms share one file as the single
source of truth and can never drift out of sync. This script never reads the prompt-ensemble
VALUES anyway (only get_coarse_pc59_{llm,nollm}.py do) -- the only real difference from the
LLM-arm script is COARSE_CACHE_DIR below, which points at coarse masks built from bare
class-name SAM3 queries instead of the ensemble.

--------------------------------------------------------------------------------------------
Expected project layout (all paths resolved from this script's own location):

    pc59_ablation/                              <- PROJECT_ROOT
    |-- train_unetaspp_pc59_nollm.py            <- this file
    |-- data_pc59/
    |   |-- JPEGImages/
    |   |-- SegmentationClassContext/
    |   `-- adjust_prompt_pc59.json             <- single source of truth for TARGET_CLASSES
    `-- coarse_cache_pc59_nollm/                <- output of get_coarse_pc59_nollm.py (has split.json)

Output (written under PROJECT_ROOT/weights_unetaspp_pc59_nollm_v1/):
    unetaspp_pc59_nollm_v1_best.pth
    unetaspp_pc59_nollm_v1_last.pth
    unetaspp_pc59_nollm_v1_history.json
    training_curves.png

Setup (run once on the VM, before this script -- NOT executed by this script itself):

    pip install torch torchvision                      # match this VM's CUDA build
    pip install "albumentations>=2.0" opencv-python-headless scipy
    pip install tqdm pillow matplotlib numpy

Usage:

    python train_unetaspp_pc59_nollm.py

If your data/checkpoint don't match the layout above, edit the CONFIG section below directly.
--------------------------------------------------------------------------------------------
"""

import os
import json
import math
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.models as tvm
from torchvision.models import ResNet34_Weights
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIG -- paths and fixed hyperparameters (safe as true module-level constants, no file I/O).
# Everything that depends on adjust_prompt_pc59.json (TARGET_CLASSES and everything derived from
# it) is populated by load_target_config() inside main(), not at import time.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PC59_IMAGES_ROOT = PROJECT_ROOT / "data_pc59" / "JPEGImages"
PC59_GT_ROOT = PROJECT_ROOT / "data_pc59" / "SegmentationClassContext"
ADJUST_PROMPT_PATH = PROJECT_ROOT / "data_pc59" / "adjust_prompt_pc59.json"  # TODO: create once target classes are chosen

COARSE_CACHE_DIR = PROJECT_ROOT / "coarse_cache_pc59_nollm"  # TODO: adjust -- output of get_coarse_pc59_nollm.py
SPLIT_JSON = COARSE_CACHE_DIR / "split.json"

WEIGHTS_DIR = PROJECT_ROOT / "weights_unetaspp_pc59_nollm_v1"
BEST_CKPT = WEIGHTS_DIR / "unetaspp_pc59_nollm_v1_best.pth"
LAST_CKPT = WEIGHTS_DIR / "unetaspp_pc59_nollm_v1_last.pth"
HISTORY_JSON = WEIGHTS_DIR / "unetaspp_pc59_nollm_v1_history.json"
PLOT_PATH = WEIGHTS_DIR / "training_curves.png"

# Full 59-class list (alphabetical order) -- used to look up each TARGET class's raw pixel value
# in SegmentationClassContext/<id>.png (index = alphabetical position + 1, 0 = background/other).
PC59_ALL_CLASSES = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book",
    "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth",
    "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food",
    "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse",
    "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves",
    "sidewalk", "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood",
]
assert len(PC59_ALL_CLASSES) == 59
GT_CLASS_TO_INDEX = {name: i + 1 for i, name in enumerate(PC59_ALL_CLASSES)}  # 1..59, 0 = background/other

# PC59 images ARE VOC2010's images -- same "no fixed aspect ratio" situation as VOC2012, so
# resize/crop to a square.
IMAGE_H = 512
IMAGE_W = 512

SEED = 42
BATCH_SIZE = 4
EPOCHS = 30
WARMUP_EPOCHS = 2
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0 if os.name == "nt" else 4
PATIENCE = 8
FOCAL_GAMMA = 1.5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Populated by load_target_config() in main()
TARGET_CLASSES = None
TARGET_TO_INDEX = None
TARGET_TO_LOCAL = None
LOCAL_TO_TARGET = None
NUM_TARGETS = None
NUM_OUT_CLASSES = None
IN_CHANNELS_BASE = None
TVERSKY_PARAMS = None
BOUNDARY_LAMBDA = None

# Populated by compute_ce_weights() in main()
CE_WEIGHTS = None
BOUNDARY_LAMBDA_TENSOR = None
TVERSKY_TENSOR = None

DEVICE = None


# ============================================================================
# Setup
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch():
    global DEVICE
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", DEVICE)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))


def load_target_config():
    """Target classes: single source of truth is adjust_prompt_pc59.json. TVERSKY_PARAMS /
    BOUNDARY_LAMBDA default to neutral values (tune per class once you know which classes were
    chosen -- see the TODO note printed below)."""
    global TARGET_CLASSES, TARGET_TO_INDEX, TARGET_TO_LOCAL, LOCAL_TO_TARGET
    global NUM_TARGETS, NUM_OUT_CLASSES, IN_CHANNELS_BASE, TVERSKY_PARAMS, BOUNDARY_LAMBDA

    with open(ADJUST_PROMPT_PATH, encoding="utf-8") as f:
        adjust_prompt = json.load(f)

    TARGET_CLASSES = list(adjust_prompt.keys())
    TARGET_TO_INDEX = {name: GT_CLASS_TO_INDEX[name] for name in TARGET_CLASSES}
    TARGET_TO_LOCAL = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
    LOCAL_TO_TARGET = {v: k for k, v in TARGET_TO_LOCAL.items()}
    NUM_TARGETS = len(TARGET_CLASSES)
    NUM_OUT_CLASSES = NUM_TARGETS + 1  # ch0 = other/bg
    assert NUM_TARGETS > 0, f"TARGET_CLASSES is empty -- create {ADJUST_PROMPT_PATH} first."

    IN_CHANNELS_BASE = 3 + NUM_TARGETS

    # TODO: tune per class once you know which classes were chosen and have inspected
    # get_coarse_pc59_llm.py's class_pixel_stats.json + a few epochs of training curves --
    # small/slender/easily-confused classes generally want higher beta (favor recall) and higher
    # boundary lambda. Defaults below are a neutral starting point (balanced Tversky, uniform
    # boundary weight) so this script is immediately runnable regardless of TARGET_CLASSES.
    TVERSKY_PARAMS = {c: (0.5, 0.5) for c in TARGET_CLASSES}
    BOUNDARY_LAMBDA = {c: 1.0 for c in TARGET_CLASSES}

    print(f"Target classes (local 1..{NUM_TARGETS}): {TARGET_CLASSES}")
    print("Output classes: other/bg(0) + " + ", ".join(f"{c}({i})" for i, c in enumerate(TARGET_CLASSES, 1)))
    print(f"Image size    : {IMAGE_H} x {IMAGE_W}  (H x W)")
    print(f"IN_CHANNELS   : {IN_CHANNELS_BASE}  (RGB + {NUM_TARGETS} coarse, no DINOv2 at this ablation level)")
    print(f"OUT_CLASSES   : {NUM_OUT_CLASSES}  (other/bg + {NUM_TARGETS} target)")
    print(f"Batch size    : {BATCH_SIZE}   Epochs: {EPOCHS} (warmup={WARMUP_EPOCHS} + cosine)   Patience: {PATIENCE}")
    print(f"Tversky params (per class): {TVERSKY_PARAMS}")
    print(f"Boundary lambda: {BOUNDARY_LAMBDA}")
    print(f"Focal gamma    : {FOCAL_GAMMA}")


# ============================================================================
# Split loading + validation
# ============================================================================

def _coarse_ok(img_id):
    p = COARSE_CACHE_DIR / f"{img_id}.npz"
    if not p.exists():
        return False
    try:
        with np.load(p) as d:
            return set(TARGET_CLASSES) - set(d.files) == set()
    except Exception:
        return False


def check_ready(img_id):
    img_path = PC59_IMAGES_ROOT / f"{img_id}.jpg"
    if not img_path.exists():
        return False, "missing_image"
    if not _coarse_ok(img_id):
        return False, "missing_or_bad_coarse_npz"
    return True, "ok"


def filter_ready(image_ids, tag):
    ok_list, skipped = [], []
    for img_id in tqdm(image_ids, desc=f"Validate {tag}"):
        ok, reason = check_ready(img_id)
        (ok_list if ok else skipped).append(img_id if ok else (img_id, reason))
    print(f"  {tag}: usable {len(ok_list)}/{len(image_ids)}")
    for item in skipped[:5]:
        print(f"    skipped: {item}")
    return ok_list


def get_gt_mask_path(img_id):
    p = PC59_GT_ROOT / f"{img_id}.png"
    return p if p.exists() else None


def load_pc59_mask(mask_path):
    """Raw PC59 label map (uint8), 0-59, no void marker -- local id 1..NUM_TARGETS will be
    derived from this in Dataset.__getitem__, everything else is other/bg=0."""
    return np.array(Image.open(mask_path), dtype=np.uint8)


def filter_has_gt(image_ids, tag):
    ok, missing = [], []
    for img_id in image_ids:
        (ok if get_gt_mask_path(img_id) is not None else missing).append(img_id)
    print(f"{tag}: GT found {len(ok)}/{len(image_ids)}  (missing {len(missing)})")
    return ok


def load_split():
    """Load internal train/val split (produced by get_coarse_pc59_llm.py) and validate against
    disk (image + GT + coarse cache with all TARGET_CLASSES keys present)."""
    with open(SPLIT_JSON) as f:
        split = json.load(f)

    internal_train_ids = split["train"]
    internal_val_ids = split["val"]
    print(f"Internal train (raw from split.json): {len(internal_train_ids)}")
    print(f"Internal val   (raw from split.json): {len(internal_val_ids)}")

    internal_train_ids = filter_ready(internal_train_ids, "internal-train")
    internal_val_ids = filter_ready(internal_val_ids, "internal-val")

    if not internal_train_ids:
        raise RuntimeError("No usable internal-train images. Check PC59_IMAGES_ROOT / COARSE_CACHE_DIR paths.")

    internal_train_ids = filter_has_gt(internal_train_ids, "internal-train")
    internal_val_ids = filter_has_gt(internal_val_ids, "internal-val")

    return internal_train_ids, internal_val_ids


# ============================================================================
# CE class weights
# ============================================================================

def compute_ce_weights(internal_train_ids):
    """Scan GT on internal-TRAIN to compute CE class weights ((N+1)-way)."""
    global CE_WEIGHTS, BOUNDARY_LAMBDA_TENSOR, TVERSKY_TENSOR

    cls_pixels = {c: 0 for c in range(NUM_OUT_CLASSES)}
    per_class_image_count = {c: 0 for c in TARGET_CLASSES}
    missing_gt = []

    for img_id in tqdm(internal_train_ids, desc="Scan GT for CE weights"):
        mask_path = get_gt_mask_path(img_id)
        if mask_path is None:
            missing_gt.append(img_id)
            continue

        label_mask = load_pc59_mask(mask_path)
        v_total = int(label_mask.size)  # every pixel is valid -- no void marker in PC59 raw masks

        target_total_px = 0
        for cls_name in TARGET_CLASSES:
            idx = TARGET_TO_INDEX[cls_name]
            local_id = TARGET_TO_LOCAL[cls_name]
            cls_mask = (label_mask == idx)
            p = int(cls_mask.sum())
            cls_pixels[local_id] += p
            target_total_px += p
            if p > 0:
                per_class_image_count[cls_name] += 1

        cls_pixels[0] += v_total - target_total_px  # other/bg = every remaining pixel

    if missing_gt:
        print(f"[warn] {len(missing_gt)} images had no matching GT mask found:")
        for m in missing_gt[:5]:
            print(f"  {m}")

    print(f"\nInternal-train images used: {len(internal_train_ids)}")
    print("Per-class image presence count:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:12s}: {per_class_image_count[cls]:5d} images")

    print("\nPixel counts (in internal-train set):")
    total = sum(cls_pixels.values())
    for c in range(NUM_OUT_CLASSES):
        name = "other/bg" if c == 0 else LOCAL_TO_TARGET[c]
        pct = 100.0 * cls_pixels[c] / max(total, 1)
        print(f"  ch {c} ({name:12s}): {cls_pixels[c]:>13,d} px  ({pct:5.2f}%)")

    freq = np.array([cls_pixels[c] for c in range(NUM_OUT_CLASSES)], dtype=np.float64)
    inv = 1.0 / np.clip(freq, 1, None)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.2, 5.0)
    inv = inv / inv.mean()

    # TODO (optional): manually boost a specific hard class here, e.g.
    #   inv[TARGET_TO_LOCAL["some_class"]] *= 1.5; inv = inv / inv.mean()
    # once you've identified (from the weights below, or early training curves) that one target
    # class needs extra help beyond inverse-frequency weighting alone.

    CE_WEIGHTS = torch.tensor(inv, dtype=torch.float32, device=DEVICE)

    print(f"\nCE class weights (clipped [0.2, 5.0], mean=1):")
    for c in range(NUM_OUT_CLASSES):
        name = "other/bg" if c == 0 else LOCAL_TO_TARGET[c]
        print(f"  ch {c} ({name:12s}): weight = {CE_WEIGHTS[c].item():.3f}")

    BOUNDARY_LAMBDA_TENSOR = torch.tensor(
        [BOUNDARY_LAMBDA[c] for c in TARGET_CLASSES], dtype=torch.float32, device=DEVICE)
    TVERSKY_TENSOR = torch.tensor(
        [TVERSKY_PARAMS[c] for c in TARGET_CLASSES], dtype=torch.float32, device=DEVICE)


# ============================================================================
# Dataset
# ============================================================================

def build_augment_no_hflip():
    return A.Compose([
        A.RandomScale(scale_limit=(-0.25, 0.25), p=0.7),
        A.PadIfNeeded(min_height=IMAGE_H, min_width=IMAGE_W,
                      border_mode=cv2.BORDER_CONSTANT, value=0,
                      mask_value=0, p=1.0),
        A.RandomCrop(height=IMAGE_H, width=IMAGE_W, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.6),
    ])


def compute_boundary_weight(target_np, alpha=1.5):
    gt = target_np.astype(bool)
    if not gt.any() or gt.all():
        return np.ones(target_np.shape, dtype=np.float32)
    d_in = distance_transform_edt(gt)
    d_out = distance_transform_edt(~gt)
    dist = (d_in + d_out).astype(np.float32)
    w = 1.0 / (dist + 1.0) ** alpha
    w_max = w.max()
    return (w / w_max).astype(np.float32) if w_max > 0 else w


class SAM3PC59Dataset(Dataset):
    """
    Each sample = 1 PC59 image (train or internal-val).

    Returns:
      input_base   : [3+NUM_TARGETS, H, W] = RGB(3) + NUM_TARGETS coarse (augmented if aug != None)
      target_label : [H, W] long -- 0..NUM_TARGETS (0 = other/bg, derived directly from GT)
      valid        : [1, H, W] float -- all-ones here (PC59 raw masks have no void/255 pixels)
      coarses      : [NUM_TARGETS, H, W] float -- for baseline (coarse) IoU comparison
      bnd_w        : [NUM_TARGETS, H, W] float -- boundary weight map per target class
      image_id     : str
    """
    def __init__(self, image_ids, coarse_dir, aug=None):
        self.image_ids = image_ids
        self.coarse_dir = Path(coarse_dir)
        self.aug = aug

    def __len__(self):
        return len(self.image_ids)

    def _load_coarses(self, img_id):
        with np.load(self.coarse_dir / f"{img_id}.npz") as d:
            arrs = [d[cls].astype(np.uint8) for cls in TARGET_CLASSES]
        return np.stack(arrs, axis=0)

    def _build_label_map(self, label_mask):
        """derive directly from raw PC59 label mask -- NOT cached.
        label = 0 (other/bg) by default; assign local id 1..NUM_TARGETS for target classes."""
        label = np.zeros(label_mask.shape, dtype=np.uint8)
        for cls_name in TARGET_CLASSES:
            idx = TARGET_TO_INDEX[cls_name]
            local_id = TARGET_TO_LOCAL[cls_name]
            label[label_mask == idx] = local_id
        return label

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        image_np = np.array(Image.open(PC59_IMAGES_ROOT / f"{img_id}.jpg").convert("RGB"))
        mask_path = get_gt_mask_path(img_id)
        label_mask = load_pc59_mask(mask_path)                                # (Ho, Wo) 0-59
        coarses_np = self._load_coarses(img_id)                               # (NUM_TARGETS, Ho, Wo) uint8

        # Resize everything to IMAGE_H x IMAGE_W
        image_np = cv2.resize(image_np, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
        label_mask = cv2.resize(label_mask, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
        coarses_resized = [
            cv2.resize(coarses_np[c], (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_NEAREST)
            for c in range(NUM_TARGETS)
        ]
        coarses_np = np.stack(coarses_resized, axis=0)

        valid_np = np.ones(label_mask.shape, dtype=np.uint8)  # no void pixels in PC59 raw masks
        label_np = self._build_label_map(label_mask)

        # Augment: RGB + masks (label + valid + coarses).
        masks_to_aug = [label_np, valid_np] + [coarses_np[c] for c in range(NUM_TARGETS)]
        if self.aug is not None:
            if random.random() < 0.5:
                image_np = image_np[:, ::-1].copy()
                for i in range(len(masks_to_aug)):
                    masks_to_aug[i] = masks_to_aug[i][:, ::-1].copy()

            out = self.aug(image=image_np, masks=masks_to_aug)
            image_np = out["image"]
            aug_masks = out["masks"]
            label_np, valid_np = aug_masks[0], aug_masks[1]
            coarses_np = np.stack(aug_masks[2:2 + NUM_TARGETS], axis=0)

        label_np = np.asarray(label_np, dtype=np.uint8)
        valid_np = (np.asarray(valid_np) > 0).astype(np.float32)
        coarses_np = (np.asarray(coarses_np) > 0).astype(np.float32)

        bnd_w = np.zeros((NUM_TARGETS, IMAGE_H, IMAGE_W), dtype=np.float32)
        for c in range(NUM_TARGETS):
            local_id = c + 1
            cls_bin = (label_np == local_id).astype(np.uint8)
            bnd_w[c] = compute_boundary_weight(cls_bin)

        rgb = TF.normalize(TF.to_tensor(image_np), IMAGENET_MEAN, IMAGENET_STD)
        coarses_t = torch.from_numpy(coarses_np)
        input_base = torch.cat([rgb, coarses_t], dim=0)

        return {
            "input_base": input_base,
            "target_label": torch.from_numpy(label_np.astype(np.int64)),
            "valid": torch.from_numpy(valid_np).unsqueeze(0),
            "coarses": coarses_t,
            "bnd_w": torch.from_numpy(bnd_w),
            "image_id": img_id,
        }


def build_datasets(internal_train_ids, internal_val_ids):
    train_ds = SAM3PC59Dataset(internal_train_ids, COARSE_CACHE_DIR, aug=build_augment_no_hflip())
    val_ds = SAM3PC59Dataset(internal_val_ids, COARSE_CACHE_DIR, aug=None)

    print(f"Train samples (internal-train, WITH aug): {len(train_ds)}")
    print(f"Val   samples (internal-val,   NO  aug ): {len(val_ds)}")

    b = train_ds[0]
    assert b["input_base"].shape == (IN_CHANNELS_BASE, IMAGE_H, IMAGE_W)
    assert b["target_label"].shape == (IMAGE_H, IMAGE_W)
    assert b["bnd_w"].shape == (NUM_TARGETS, IMAGE_H, IMAGE_W)
    print(f"Sample[0]: image_id={b['image_id']}")
    print(f"  input_base   : {tuple(b['input_base'].shape)}  (RGB+{NUM_TARGETS} coarse)")
    print(f"  target_label : {tuple(b['target_label'].shape)}  unique: {sorted(b['target_label'].unique().tolist())}")
    print(f"  valid        : {tuple(b['valid'].shape)}  fg={float(b['valid'].mean()):.4f}")
    print(f"  coarses      : {tuple(b['coarses'].shape)}  fg per-class: {[round(float(b['coarses'][c].mean()), 4) for c in range(NUM_TARGETS)]}")
    print(f"  bnd_w        : {tuple(b['bnd_w'].shape)}  range: [{float(b['bnd_w'].min()):.3f}, {float(b['bnd_w'].max()):.3f}]")

    return train_ds, val_ds


# ============================================================================
# Model: ResNet-34 + ASPP + UNet decoder (no DINOv2)
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
        return self.project(torch.cat(
            [self.b1(x), self.b6(x), self.b12(x), self.b18(x), gap], dim=1))


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
    """
    ResNet-34 + ASPP + UNet decoder -- SAM3+UNet+ASPP ablation level (no DINOv2 branch).

    Input:
      - input_base : [B, 3+NUM_TARGETS, H, W]  = RGB(3) + NUM_TARGETS coarse   (H=W=512)

    First conv init:
      - ch 0-2                     : ImageNet pretrained RGB weights
      - ch 3 .. in_channels_base-1 : 0  (NUM_TARGETS coarse channels)
    """
    def __init__(self, in_channels_base=9, out_channels=7):
        super().__init__()
        self.in_channels_base = in_channels_base

        backbone = tvm.resnet34(weights=ResNet34_Weights.DEFAULT)

        orig_w = backbone.conv1.weight.data.clone()  # [64, 3, 7, 7]
        new_conv = nn.Conv2d(in_channels_base, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = orig_w
            new_conv.weight[:, 3:] = 0.0
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

    def forward(self, input_base):
        H, W = input_base.shape[-2:]

        e0 = self.enc0(input_base)
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
        return self.head(d)  # [B, NUM_OUT_CLASSES, H, W]


def build_model():
    model = ResNet34UNetASPPPC59(
        in_channels_base=IN_CHANNELS_BASE,
        out_channels=NUM_OUT_CLASSES,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("ResNet34UNetASPPPC59 (SAM3+UNet+ASPP ablation, no DINOv2)")
    print(f"  input_base channels: {IN_CHANNELS_BASE}  (RGB + {NUM_TARGETS} coarse)")
    print(f"  out_classes        : {NUM_OUT_CLASSES}")
    print(f"  Image size         : {IMAGE_H} x {IMAGE_W}")
    print(f"  Trainable params   : {n_params:,}")

    with torch.no_grad():
        _x = torch.zeros(1, IN_CHANNELS_BASE, IMAGE_H, IMAGE_W).to(DEVICE)
        _y = model(_x)
        print(f"  Forward check: input_base={tuple(_x.shape)} -> out={tuple(_y.shape)}  ok")
        assert _y.shape == (1, NUM_OUT_CLASSES, IMAGE_H, IMAGE_W)
    del _x, _y

    return model


# ============================================================================
# Loss: Focal CE + per-class Tversky + Boundary Loss
# ============================================================================

def masked_ce(logits, target_label, valid):
    ce_per_pix = F.cross_entropy(logits, target_label, weight=CE_WEIGHTS, reduction="none")
    if FOCAL_GAMMA > 0:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target_label.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(0.0, 1.0)
        focal_w = (1.0 - pt).pow(FOCAL_GAMMA)
        ce_per_pix = ce_per_pix * focal_w
    v = valid.squeeze(1)
    return (ce_per_pix * v).sum() / v.sum().clamp_min(1.0)


def per_class_loss(probs, target_label, valid, bnd_w):
    v = valid.squeeze(1).float()
    eps = 1.0
    dice_sum = torch.tensor(0.0, device=probs.device)
    bnd_sum = torch.tensor(0.0, device=probs.device)

    for c_idx in range(NUM_TARGETS):
        local_id = c_idx + 1
        p = probs[:, local_id] * v
        t = (target_label == local_id).float() * v

        alpha = TVERSKY_TENSOR[c_idx, 0]
        beta = TVERSKY_TENSOR[c_idx, 1]
        tp = (p * t).sum(dim=(1, 2))
        fp = (p * (1 - t)).sum(dim=(1, 2))
        fn = ((1 - p) * t).sum(dim=(1, 2))
        tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        cls_loss = (1.0 - tv).mean()
        dice_sum = dice_sum + cls_loss

        p_c = torch.clamp(p, 1e-7, 1.0 - 1e-7)
        bce_pix = -(t * torch.log(p_c) + (1.0 - t) * torch.log(1.0 - p_c))
        wmap = bnd_w[:, c_idx] * v
        bnd_c = (bce_pix * wmap).sum() / wmap.sum().clamp_min(1.0)
        bnd_sum = bnd_sum + BOUNDARY_LAMBDA_TENSOR[c_idx] * bnd_c

    dice_avg = dice_sum / NUM_TARGETS
    bnd_avg = bnd_sum / NUM_TARGETS
    return dice_avg, bnd_avg


def combined_loss(logits, target_label, valid, bnd_w):
    ce = masked_ce(logits, target_label, valid)
    probs = F.softmax(logits, dim=1)
    dice_loss, bnd_loss = per_class_loss(probs, target_label, valid, bnd_w)
    total = 0.5 * ce + 0.3 * dice_loss + 0.2 * bnd_loss
    return total, ce.detach(), dice_loss.detach(), bnd_loss.detach()


# ============================================================================
# Metrics
# ============================================================================

def init_metric_dict():
    return {cls: {"tp": 0, "fp": 0, "fn": 0, "c_tp": 0, "c_fp": 0, "c_fn": 0} for cls in TARGET_CLASSES}


def update_metrics_multi(mdict, pred_label, target_label, valid_bool, coarses_bool):
    for c_idx, cls in enumerate(TARGET_CLASSES):
        local_id = c_idx + 1
        v = valid_bool
        p = (pred_label == local_id) & v
        t = (target_label == local_id) & v
        c = coarses_bool[:, c_idx] & v
        mdict[cls]["tp"] += int((p & t).sum().item())
        mdict[cls]["fp"] += int((p & ~t).sum().item())
        mdict[cls]["fn"] += int((~p & t).sum().item())
        mdict[cls]["c_tp"] += int((c & t).sum().item())
        mdict[cls]["c_fp"] += int((c & ~t).sum().item())
        mdict[cls]["c_fn"] += int((~c & t).sum().item())


def iou_from(tp, fp, fn):
    d = tp + fp + fn
    return float(tp / d) if d > 0 else float("nan")


def summarize(mdict):
    per_iou = {cls: iou_from(mdict[cls]["tp"], mdict[cls]["fp"], mdict[cls]["fn"]) for cls in TARGET_CLASSES}
    per_coarse = {cls: iou_from(mdict[cls]["c_tp"], mdict[cls]["c_fp"], mdict[cls]["c_fn"]) for cls in TARGET_CLASSES}
    miou = float(np.nanmean(list(per_iou.values())))
    coarse_miou = float(np.nanmean(list(per_coarse.values())))
    return {"per_iou": per_iou, "per_coarse": per_coarse,
            "miou": miou, "coarse_miou": coarse_miou, "delta": miou - coarse_miou}


# ============================================================================
# DataLoader + Optimizer + Scheduler
# ============================================================================

def build_dataloaders_and_optimizer(train_ds, val_ds, model):
    pin = DEVICE.type == "cuda"
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin, drop_last=False)
    if NUM_WORKERS > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)

    print(f"Train batches: {len(train_loader)}  ({len(train_ds)} images, WITH aug)")
    print(f"Val   batches: {len(val_loader)}    ({len(val_ds)} images, NO  aug -- real internal-val, not train self-eval)")

    _b = next(iter(train_loader))
    print(f"Batch input_base   : {tuple(_b['input_base'].shape)}    expect [B,{IN_CHANNELS_BASE},{IMAGE_H},{IMAGE_W}]")
    print(f"Batch target_label : {tuple(_b['target_label'].shape)}  expect [B,{IMAGE_H},{IMAGE_W}]")
    print(f"Batch bnd_w        : {tuple(_b['bnd_w'].shape)}        expect [B,{NUM_TARGETS},{IMAGE_H},{IMAGE_W}]")
    assert _b["input_base"].shape[1] == IN_CHANNELS_BASE
    del _b

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast_ctx():
        return torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()

    print(f"Optimizer : AdamW  lr={LR}  wd={WEIGHT_DECAY}")
    print(f"Scheduler : Linear warmup ({WARMUP_EPOCHS} ep) -> CosineAnnealing -> 0.05*LR")
    print(f"AMP       : {use_amp}")

    return train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx


# ============================================================================
# Training loop
# ============================================================================

def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx):
    history = []
    best_miou = -1.0
    best_epoch = -1
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        # -- Train --
        model.train()
        tr_loss = tr_ce = tr_dice = tr_bnd = 0.0
        tr_n = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [train]", leave=False)
        for batch in bar:
            inp_base = batch["input_base"].to(DEVICE, non_blocking=True)
            tgt_label = batch["target_label"].to(DEVICE, non_blocking=True)
            valid = batch["valid"].to(DEVICE, non_blocking=True)
            bnd_w = batch["bnd_w"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                logits = model(inp_base)
                loss, ce_d, dice_d, bnd_d = combined_loss(logits, tgt_label, valid, bnd_w)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = inp_base.size(0)
            tr_loss += loss.item() * bs
            tr_ce += ce_d.item() * bs
            tr_dice += dice_d.item() * bs
            tr_bnd += bnd_d.item() * bs
            tr_n += bs

            bar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_d.item():.4f}",
                             dice=f"{dice_d.item():.4f}", bnd=f"{bnd_d.item():.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # -- Eval on the internal-val --
        model.eval()
        mdict = init_metric_dict()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [val  ]", leave=False):
                inp_base = batch["input_base"].to(DEVICE, non_blocking=True)
                tgt_label = batch["target_label"].to(DEVICE, non_blocking=True)
                valid = batch["valid"].to(DEVICE, non_blocking=True)
                coarses = batch["coarses"].to(DEVICE, non_blocking=True)
                with autocast_ctx():
                    logits = model(inp_base)
                pred_label = logits.argmax(dim=1)
                valid_bool = valid.squeeze(1) >= 0.5
                coarses_bool = coarses >= 0.5
                update_metrics_multi(mdict, pred_label, tgt_label, valid_bool, coarses_bool)

        n = max(tr_n, 1)
        s = summarize(mdict)
        rec = dict(epoch=epoch,
                   train_loss=tr_loss / n, ce=tr_ce / n, dice=tr_dice / n, bnd=tr_bnd / n,
                   miou=s["miou"], coarse_miou=s["coarse_miou"], delta=s["delta"],
                   lr=float(optimizer.param_groups[0]["lr"]),
                   per_iou={k: float(v) for k, v in s["per_iou"].items()},
                   per_coarse={k: float(v) for k, v in s["per_coarse"].items()})
        history.append(rec)

        print(f"\nEpoch {epoch:2d}/{EPOCHS}")
        print(f"  Loss: {tr_loss/n:.4f}  (ce={tr_ce/n:.4f}  dice={tr_dice/n:.4f}  bnd={tr_bnd/n:.4f})")
        print(f"  [internal-val] Coarse mIoU: {s['coarse_miou']:.4f}  ->  Refined mIoU: {s['miou']:.4f}  (delta {s['delta']:+.4f})")
        print("  Per-class (coarse -> refined), internal-val:")
        for cls in TARGET_CLASSES:
            ci = s["per_coarse"][cls]
            ri = s["per_iou"][cls]
            print(f"    {cls:12s}: {ci:.4f} -> {ri:.4f}  (delta {ri-ci:+.4f})")

        ckpt = dict(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=epoch,
            miou=s["miou"],
            coarse_miou=s["coarse_miou"],
            target_classes=TARGET_CLASSES,
            target_to_index=TARGET_TO_INDEX,
            target_to_local=TARGET_TO_LOCAL,
            boundary_lambda=BOUNDARY_LAMBDA,
            in_channels_base=IN_CHANNELS_BASE,
            num_out_classes=NUM_OUT_CLASSES,
            image_h=IMAGE_H,
            image_w=IMAGE_W,
            ce_weights=CE_WEIGHTS.detach().cpu().numpy().tolist(),
            tversky_params=TVERSKY_PARAMS,
            focal_gamma=FOCAL_GAMMA,
            arch_version="multi_class_softmax_pc59_unetaspp_nollm_v1",
        )
        torch.save(ckpt, LAST_CKPT)

        if s["miou"] > best_miou:
            best_miou = s["miou"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(ckpt, BEST_CKPT)
            print(f"  BEST checkpoint (epoch {epoch}  mIoU={best_miou:.4f})")
        else:
            epochs_since_best += 1
            print(f"  no improvement ({epochs_since_best}/{PATIENCE} since best epoch {best_epoch}, mIoU={best_miou:.4f})")

        with open(HISTORY_JSON, "w") as f:
            json.dump(history, f, indent=2)

        if epochs_since_best >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no internal-val mIoU improvement for {PATIENCE} epochs).")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training done | Best epoch: {best_epoch} | Best internal-val mIoU: {best_miou:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Best ckpt : {BEST_CKPT}")
    print(f"Last ckpt : {LAST_CKPT}")

    return history, best_epoch, best_miou


# ============================================================================
# Plot
# ============================================================================

def plot_curves(history, best_epoch, best_miou, out_path):
    ep = [r["epoch"] for r in history]
    tr_loss = [r["train_loss"] for r in history]
    ce_v = [r["ce"] for r in history]
    dice_v = [r["dice"] for r in history]
    bnd_v = [r["bnd"] for r in history]
    miou = [r["miou"] for r in history]
    cmiou = [r["coarse_miou"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ep, tr_loss, "o-", label="Total")
    axes[0].plot(ep, ce_v, "s--", label="Focal CE", alpha=0.7)
    axes[0].plot(ep, dice_v, "^--", label="Tversky/Dice", alpha=0.7)
    axes[0].plot(ep, bnd_v, "d--", label="Boundary", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Losses")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, cmiou, "o-", label="Coarse (SAM3)", color="#d9534f")
    axes[1].plot(ep, miou, "s-", label="Refined (UNet+ASPP)", color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title(f"Internal-val mIoU ({NUM_TARGETS}-class mean)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    for cls in TARGET_CLASSES:
        pi = [r["per_iou"][cls] for r in history]
        axes[2].plot(ep, pi, "o-", label=cls, alpha=0.8)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("Per-class IoU (refined, internal-val)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"UNet+ASPP PC59 (no DINOv2) -- Best Epoch {best_epoch} | Internal-val mIoU {best_miou:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    setup_torch()
    set_seed(SEED)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    load_target_config()
    internal_train_ids, internal_val_ids = load_split()
    compute_ce_weights(internal_train_ids)
    train_ds, val_ds = build_datasets(internal_train_ids, internal_val_ids)
    model = build_model()
    train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx = \
        build_dataloaders_and_optimizer(train_ds, val_ds, model)

    history, best_epoch, best_miou = train_model(
        model, train_loader, val_loader, optimizer, scheduler, scaler, autocast_ctx)

    if history:
        plot_curves(history, best_epoch, best_miou, PLOT_PATH)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] train_unetaspp_pc59_nollm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
