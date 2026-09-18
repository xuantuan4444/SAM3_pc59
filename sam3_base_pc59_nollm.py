#!/usr/bin/env python3
"""
sam3_base_pc59_nollm.py
========================
Standalone script version of `sam3-base-pc59-nollm.ipynb`.

Evaluates SAM3 (no-LLM, plain class-name prompts) on the PASCAL-Context 59-class val split
(5105 images) — the baseline eval used to rank classes and pick the target-class set for the
refine-network ablation study. Logic is unchanged from the notebook; only I/O has been adapted
for a plain VM with no Jupyter (`plt.show()`/`display()` -> saved PNG/CSV files, paths ->
PROJECT_ROOT-relative). One bug present in the uploaded notebook (Cell "Bar chart" referenced
an undefined `NUM_FG_CLASSES`, left over from an earlier edit) is fixed here.

--------------------------------------------------------------------------------------------
Expected project layout (all paths resolved from this script's own location):

    sam3_base_pc59_nollm/
    ├── sam3_base_pc59_nollm.py           <- this file
    ├── data_pc59/
    │   ├── JPEGImages/                    <- VOC2010 images, <id>.jpg
    │   ├── SegmentationClassContext/      <- 59-class PNG masks (output of prepare_pc59_mat_to_png)
    │   ├── 59_labels.txt
    │   └── pascal_context_val.txt         <- "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png" per line
    └── weight_sam3/sam3.pt

Output (written under this script's folder):
    per_class_iou_bar_chart.png
    class_visualizations/<class_name>.png   (up to 2 example images per class present in val)
    summary_metrics.csv
    per_class_metrics.csv

Setup (run once on the VM, before this script — NOT executed by this script itself):

    pip install torch torchvision                      # match this VM's CUDA build
    pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
    pip install iopath ftfy portalocker pandas matplotlib pillow numpy tqdm

Usage:

    python sam3_base_pc59_nollm.py

If your data/checkpoint don't match the layout above, edit the CONFIG section below directly.
--------------------------------------------------------------------------------------------
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# CONFIG — every path is resolved from PROJECT_ROOT (this script's own folder), never from the
# current working directory and never from an environment variable.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PC59_IMAGES_ROOT = PROJECT_ROOT / "data_pc59" / "JPEGImages"
PC59_GT_ROOT = PROJECT_ROOT / "data_pc59" / "SegmentationClassContext"
LABELS_TXT_PATH = PROJECT_ROOT / "data_pc59" / "59_labels.txt"
VAL_FILE = PROJECT_ROOT / "data_pc59" / "pascal_context_val.txt"
SAM3_CKPT = PROJECT_ROOT / "weight_sam3" / "sam3.pt"

OUT_BAR_CHART = PROJECT_ROOT / "per_class_iou_bar_chart.png"
OUT_VIS_DIR = PROJECT_ROOT / "class_visualizations"
OUT_SUMMARY_CSV = PROJECT_ROOT / "summary_metrics.csv"
OUT_CLASS_CSV = PROJECT_ROOT / "per_class_metrics.csv"

VOID_VALUE = 255
# Confirmed from the .mat->PNG conversion script's own LUT logic (prepare_pc59_mat_to_png):
# 0 = background, 1-59 = class (alphabetical order).
BACKGROUND_IS_ZERO = True
CONFIDENCE_THRESHOLD = 0.3
MAX_VAL_IMAGES = None  # None = all val images; set an int for a quick smoke-test run
NUM_SAMPLE_VIS = 2      # up to N example images stored per class for visualization

# No-LLM arm: every class queries SAM3 with its own bare name.
CLASS_PROMPTS_OVERRIDE = {}

# Populated in main() via load_classes_and_split() / build_model()
PC59_CLASSES = None
NUM_CLASSES = None
INDEX_TO_CLASS = None
DEVICE = None
processor = None


# ============================================================================
# Label parsing / GT normalization
# ============================================================================

def parse_pc59_labels(path):
    """Robust parser for 59_labels.txt -- tries 'idx: name' and 'idx name'; falls back to
    1-indexed line order if a line has no explicit leading index."""
    id_to_name = {}
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line_no, line in enumerate(lines, start=1):
        idx, name = None, None
        if ":" in line:
            left, right = line.split(":", 1)
            if left.strip().isdigit():
                idx, name = int(left.strip()), right.strip()
        if idx is None:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                idx, name = int(parts[0]), parts[1].strip()
        if idx is None:
            idx, name = line_no, line
        id_to_name[idx] = name
    return id_to_name


def normalize_gt(raw_mask):
    """Convert the raw GT mask into this script's canonical form: 0..58 = class (0-indexed),
    255 = ignore/background."""
    if BACKGROUND_IS_ZERO:
        out = raw_mask.astype(np.int16) - 1
        out[out < 0] = VOID_VALUE  # raw 0 (background) wraps to ignore
        return out.astype(np.uint8)
    return raw_mask


def load_pc59_mask(mask_path):
    return np.array(Image.open(mask_path), dtype=np.uint8)


# ============================================================================
# SAM3 prediction helpers
# ============================================================================

def extract_masks_from_state(state):
    """Read masks & scores directly from SAM3 state dict.

    After set_text_prompt():
      state["masks"]  -> bool tensor [N, 1, H, W] (already resized to original image)
      state["scores"] -> float tensor [N]
    """
    if "masks" not in state or "scores" not in state:
        return [], []
    masks_tensor = state["masks"]
    scores_tensor = state["scores"]
    if masks_tensor is None or len(masks_tensor) == 0:
        return [], []
    out_masks, out_scores = [], []
    for i in range(len(masks_tensor)):
        out_masks.append(masks_tensor[i].squeeze(0).cpu().numpy())
        out_scores.append(scores_tensor[i].item())
    return out_masks, out_scores


def fast_confusion_matrix(gt, pred, num_classes):
    """Compute confusion matrix, ignoring VOID_VALUE."""
    valid = (gt != VOID_VALUE) & (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(
        num_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64),
        minlength=num_classes * num_classes,
    )
    return hist.reshape(num_classes, num_classes)


def predict_semantic_map_native(image_path, class_names):
    """
    Use Sam3Processor to predict a semantic label map, no-LLM (plain class-name prompt) version.

    Returns: label_map (H, W) uint8 -- 0..58 = class index (0-indexed, matches PC59_CLASSES
    directly). Pixels with no confident detection from ANY class default to VOID_VALUE (255) --
    NOT 0, since 0 is a real class ("aeroplane") in this label space.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    state = processor.set_image(image)

    pred_map = np.full((height, width), VOID_VALUE, dtype=np.uint8)
    best_score = np.zeros((height, width), dtype=np.float32)

    for class_idx, class_name in enumerate(class_names):  # 0-indexed, no offset
        prompts = CLASS_PROMPTS_OVERRIDE.get(class_name, [class_name])
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            masks, scores = extract_masks_from_state(state)
            for mask, score in zip(masks, scores):
                update = mask & (score > best_score)
                pred_map[update] = class_idx
                best_score[update] = score

    return pred_map


# ============================================================================
# Visualization helpers
# ============================================================================

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
    colored[mask == VOID_VALUE] = [128, 128, 128]  # void -> gray
    return colored


# ============================================================================
# Setup
# ============================================================================

def setup_torch():
    global DEVICE
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", DEVICE)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))


def build_model():
    global processor
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _sam3_dir = Path(sam3.__file__).parent
    bpe_path = _sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=str(bpe_path), checkpoint_path=str(SAM3_CKPT), load_from_HF=False)
    model = model.to(DEVICE)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    print(f"Model loaded. Confidence threshold: {CONFIDENCE_THRESHOLD}")


def load_classes_and_split():
    global PC59_CLASSES, NUM_CLASSES, INDEX_TO_CLASS

    id_to_name = parse_pc59_labels(LABELS_TXT_PATH)
    PC59_CLASSES = [id_to_name[i] for i in sorted(id_to_name)]
    NUM_CLASSES = len(PC59_CLASSES)
    assert NUM_CLASSES == 59, f"expected 59 classes, parsed {NUM_CLASSES} from {LABELS_TXT_PATH}"
    INDEX_TO_CLASS = {i: name for i, name in enumerate(PC59_CLASSES)}

    print(f"Parsed {NUM_CLASSES} classes from {LABELS_TXT_PATH.name}:")
    for i in list(range(5)) + list(range(54, 59)):
        print(f"  {i}: {PC59_CLASSES[i]}")

    if VAL_FILE.exists():
        with open(VAL_FILE, "r", encoding="utf-8") as f:
            # Each line is "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png" -- take the
            # first column and strip directory + extension down to the bare id.
            val_ids = [Path(line.strip().split()[0]).stem for line in f if line.strip()]
        print(f"Loaded val split from {VAL_FILE} ({len(val_ids)} images)")
        if len(val_ids) < 3000:
            print(f"[warn] only {len(val_ids)} val images -- expected ~5105 for the full "
                  "PASCAL-Context split; double-check VAL_FILE.")
    else:
        print(f"[warn] {VAL_FILE} not found -- falling back to every PNG in {PC59_GT_ROOT}")
        val_ids = sorted(p.stem for p in PC59_GT_ROOT.glob("*.png"))
        print(f"Fallback: {len(val_ids)} images found directly in GT folder")

    if val_ids:
        sample_mask_path = PC59_GT_ROOT / f"{val_ids[0]}.png"
        if sample_mask_path.exists():
            raw = np.array(Image.open(sample_mask_path))
            raw_uniques = np.unique(raw)
            print(f"\nSample mask '{sample_mask_path.name}' RAW unique values ({len(raw_uniques)}): {raw_uniques}")
            norm = normalize_gt(raw)
            norm_uniques = np.unique(norm)
            unexpected = sorted(set(norm_uniques.tolist()) - set(range(NUM_CLASSES)) - {VOID_VALUE})
            if unexpected:
                print(f"[warn] normalized values outside expected range: {unexpected} "
                      "-- BACKGROUND_IS_ZERO may be set wrong.")
            else:
                print("Normalized range OK (0-58=class, 255=ignore)")

    print(f"\nPC59 images : {PC59_IMAGES_ROOT}")
    print(f"PC59 GT     : {PC59_GT_ROOT}")
    print(f"Val images  : {len(val_ids)}")

    return val_ids


# ============================================================================
# Evaluation loop
# ============================================================================

def run_evaluation(val_ids):
    run_ids = val_ids if MAX_VAL_IMAGES is None else val_ids[:MAX_VAL_IMAGES]
    total = len(run_ids)

    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    errors = []

    start_time = time.time()
    pbar = tqdm(run_ids, desc="Evaluating PC59 val", dynamic_ncols=True)
    for idx, image_id in enumerate(pbar):
        image_path = PC59_IMAGES_ROOT / f"{image_id}.jpg"
        mask_path = PC59_GT_ROOT / f"{image_id}.png"

        if not image_path.exists() or not mask_path.exists():
            tqdm.write(f"[{idx+1:4d}/{total}] SKIP {image_id} -- file not found")
            continue

        t0 = time.time()
        try:
            gt = normalize_gt(load_pc59_mask(mask_path))  # raw -> canonical 0-58/255
            pred = predict_semantic_map_native(image_path, PC59_CLASSES)
            t_img = time.time() - t0

            conf_mat += fast_confusion_matrix(gt, pred, num_classes=NUM_CLASSES)

            gt_classes = set(np.unique(gt)) - {VOID_VALUE}
            gt_classes = {c for c in gt_classes if 0 <= c < NUM_CLASSES}
            for cls_idx in gt_classes:
                if len(class_samples[cls_idx]) < NUM_SAMPLE_VIS:
                    class_samples[cls_idx].append({
                        "id": image_id, "image_path": str(image_path), "gt": gt, "pred": pred,
                    })

            cls_names = [PC59_CLASSES[c] for c in sorted(gt_classes)]
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
    print(f"Done. {total} images in {elapsed/60:.2f} min ({elapsed/max(total,1):.2f} s/img)")
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

    # mIoU / mDice over all 59 classes -- no background class exists in this label space
    miou = np.nanmean(iou)
    mdice = np.nanmean(dice)

    summary_df = pd.DataFrame({
        "Metric": ["Pixel Accuracy", "mIoU", "Mean Dice"],
        "Value": [pixel_acc, miou, mdice],
    })

    class_df = pd.DataFrame({
        "Class": [INDEX_TO_CLASS[i] for i in range(NUM_CLASSES)],
        "IoU": iou[0:NUM_CLASSES],
        "Dice": dice[0:NUM_CLASSES],
        "GT Pixels": gt_sum[0:NUM_CLASSES].astype(np.int64),
        "Pred Pixels": pred_sum[0:NUM_CLASSES].astype(np.int64),
    }).sort_values("IoU", ascending=True).reset_index(drop=True)

    print("=" * 50)
    print("SUMMARY METRICS")
    print("=" * 50)
    print(summary_df.to_string(index=False, formatters={"Value": lambda x: f"{x:.4f}"}))

    print(f"\n{'='*50}\nPER-CLASS METRICS ({NUM_CLASSES} classes, sorted by IoU ascending)\n{'='*50}")
    print(class_df.to_string(index=False, formatters={"IoU": lambda x: f"{x:.4f}", "Dice": lambda x: f"{x:.4f}"}))

    print("\n" + "=" * 50 + "\n5 WEAKEST CLASSES (lowest IoU)\n" + "=" * 50)
    print(class_df.head(5).to_string(index=False))
    print("\n" + "=" * 50 + "\n5 STRONGEST CLASSES (highest IoU)\n" + "=" * 50)
    print(class_df.tail(5).to_string(index=False))

    print(f"\n>>> mIoU: {miou:.4f}")
    print(f">>> Pixel Accuracy: {pixel_acc:.4f}")
    print(f">>> Mean Dice: {mdice:.4f}")

    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    class_df.to_csv(OUT_CLASS_CSV, index=False)
    print(f"\nSaved: {OUT_SUMMARY_CSV}")
    print(f"Saved: {OUT_CLASS_CSV}")

    return iou, class_df, miou


# ============================================================================
# Plots (saved to PNG -- this VM has no display)
# ============================================================================

def plot_bar_chart(class_df, miou):
    fig, ax = plt.subplots(figsize=(10, max(7, NUM_CLASSES * 0.22)))
    colors = plt.cm.RdYlGn(class_df["IoU"].values)
    bars = ax.barh(class_df["Class"], class_df["IoU"], color=colors)
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_title(f"PC59 Val -- Per-class IoU (SAM3 Native, no-LLM prompts, mIoU={miou:.4f})", fontsize=13)
    ax.set_xlim(0, 1)
    ax.axvline(x=miou, color="blue", linestyle="--", linewidth=1, label=f"mIoU = {miou:.4f}")
    ax.legend(loc="lower right")

    for bar, val in zip(bars, class_df["IoU"].values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}",
                 va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_BAR_CHART, dpi=120)
    plt.close(fig)
    print(f"Saved: {OUT_BAR_CHART}")


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

        fig.suptitle(f"Class: {cls_name} (IoU = {iou[cls_idx]:.4f})", fontsize=14, fontweight="bold")

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
            axes[row, 2].set_title("Prediction (SAM3, no-LLM prompts)", fontsize=10)
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
    build_model()
    val_ids = load_classes_and_split()
    conf_mat, class_samples = run_evaluation(val_ids)
    iou, class_df, miou = compute_metrics(conf_mat)
    plot_bar_chart(class_df, miou)
    plot_class_visualizations(class_samples, iou)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] sam3_base_pc59_nollm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)