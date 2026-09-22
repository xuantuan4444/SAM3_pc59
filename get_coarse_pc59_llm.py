#!/usr/bin/env python3
"""
get_coarse_pc59_llm.py
=======================
Standalone script version of get_coarse_pc59_llm.ipynb.

Precomputes SAM3 coarse binary masks (LLM prompt ensemble, from adjust_prompt_pc59.json) for
the PASCAL-Context 59 ablation target classes, over the official PC59 train split (4998 images,
listed in pascal_context_train.txt), then packages the resulting cache directory into a .zip.

adjust_prompt_pc59.json does not exist yet (target classes not chosen until
sam3_baseline_pc59_nollm.py's per-class IoU ranking is available) -- this script still runs
without it (every step is a safe no-op) but the cache built now will be empty/incomplete --
re-run once the file exists.

Logic is unchanged from the notebook -- only I/O, packaging, and control flow were adapted for
unattended script execution on a plain GPU VM (no Kaggle-specific paths/magics, no notebook
display calls).

--------------------------------------------------------------------------------------------
Expected project layout (all paths below are resolved from this script's own location, not
from the current working directory or any environment variable -- run it from anywhere):

    pc59_ablation/                              <- PROJECT_ROOT (this script's parent folder)
    |-- get_coarse_pc59_llm.py                  <- this file
    |-- data_pc59/
    |   |-- JPEGImages/                          <- VOC2010 images, <id>.jpg
    |   |-- SegmentationClassContext/            <- 59-class PNG masks (output of prepare_pc59_mat_to_png)
    |   |-- pascal_context_train.txt             <- "JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png" per line
    |   `-- adjust_prompt_pc59.json               <- TODO: create once target classes are chosen
    `-- weight_sam3/
        `-- sam3.pt                              <- SAM3 checkpoint

Setup (run once on the VM, before this script -- NOT executed by this script itself):

    pip install torch torchvision                      # match this VM's CUDA build
    pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
    pip install iopath ftfy portalocker tqdm pillow matplotlib numpy

Usage (from any directory):

    python /path/to/pc59_ablation/get_coarse_pc59_llm.py [--rebuild-cache]

If your data/checkpoint don't match the layout above, edit the CONFIG section below directly.
--------------------------------------------------------------------------------------------
"""

import argparse
import gc
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no X server needed on a remote VM
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# CONFIG -- every path is resolved from PROJECT_ROOT (this script's own folder), never from the
# current working directory and never from an environment variable.
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

PC59_IMAGES_ROOT = PROJECT_ROOT / "data_pc59" / "JPEGImages"
PC59_GT_ROOT = PROJECT_ROOT / "data_pc59" / "SegmentationClassContext"
TRAIN_FILE = PROJECT_ROOT / "data_pc59" / "pascal_context_train.txt"
ADJUST_PROMPT_PATH = PROJECT_ROOT / "data_pc59" / "adjust_prompt_pc59.json"  # TODO: create once target classes are chosen
SAM3_CKPT = PROJECT_ROOT / "weight_sam3" / "sam3.pt"

COARSE_CACHE_DIR = PROJECT_ROOT / "coarse_cache_pc59"
COARSE_CACHE_ZIP = PROJECT_ROOT / "coarse_cache_pc59.zip"

COARSE_THRESHOLDS = [0.50, 0.30, 0.20, 0.15]
SEED = 42

# Full 59-class list (alphabetical order) -- used to look up each TARGET class's raw pixel value
# in SegmentationClassContext/<id>.png (index = alphabetical position + 1, 0 = background/other).
# Identical convention to prepare_pc59_mat_to_png.py and sam3_baseline_pc59_llm.py's PC59_CLASSES.
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


# ============================================================================
# Setup helpers
# ============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch() -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch :", torch.__version__)
    print("Device  :", device)
    if torch.cuda.is_available():
        print("GPU     :", torch.cuda.get_device_name(0))
    return device


def load_prompt_ensemble(adjust_prompt_path: Path):
    if adjust_prompt_path.exists():
        with open(adjust_prompt_path, encoding="utf-8") as f:
            adjust_prompt = json.load(f)
    else:
        print(f"[warn] {adjust_prompt_path} not found -- TARGET_CLASSES will be empty until you "
              "create this file. Every step below is safe to run (will just do nothing) but the "
              "cache built now will be empty/incomplete -- re-run this script after creating it.")
        adjust_prompt = {}
    target_classes = list(adjust_prompt.keys())
    class_prompts = [adjust_prompt[c] for c in target_classes]
    return target_classes, class_prompts


def discover_pc59_train_ids(train_file: Path):
    """pascal_context_train.txt is 'JPEGImages/<id>.jpg GroundTruth_trainval_png/<id>.png' per
    line -- take the first column and reduce to the bare id."""
    with open(train_file, "r", encoding="utf-8") as f:
        ids = [Path(line.strip().split()[0]).stem for line in f if line.strip()]
    return ids


def split_internal_train_val(all_ids, cache_dir: Path, seed: int = SEED, val_ratio: float = 0.10):
    """Carve an internal train/val split out of the official PC59 train set (10% val)."""
    rng = random.Random(seed)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    internal_val_ids = shuffled[:n_val]
    internal_train_ids = shuffled[n_val:]
    print(f"Internal train: {len(internal_train_ids)} | Internal val: {len(internal_val_ids)}")

    split_path = cache_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump({"train": internal_train_ids, "val": internal_val_ids}, f)
    print(f"Split saved to {split_path}")
    return internal_train_ids, internal_val_ids


# ============================================================================
# GT mask helpers
# ============================================================================

def get_gt_mask_path(img_id: str, gt_root: Path):
    p = gt_root / f"{img_id}.png"
    return p if p.exists() else None


def load_pc59_mask(mask_path: Path):
    """Raw PC59 label map (uint8): 0=background/other, 1-59=class (alphabetical). Unlike
    VOC/Cityscapes, there is no genuine void/255 value in this raw file -- every pixel is
    "valid"."""
    return np.array(Image.open(mask_path), dtype=np.uint8)


# ============================================================================
# SAM3 coarse cache precomputation
# ============================================================================

def _extract_masks_scores(state):
    if "masks" not in state or state["masks"] is None or len(state["masks"]) == 0:
        return [], []
    masks, scores = [], []
    for i in range(len(state["masks"])):
        masks.append(state["masks"][i].squeeze(0).detach().cpu().numpy().astype(bool))
        scores.append(float(state["scores"][i].item()))
    return masks, scores


def _union_at_threshold(masks, scores, shape_hw, thresholds):
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


def _safe_cuda_empty_cache():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[warn] torch.cuda.empty_cache failed: {e}")


def build_sam3_processor(sam3_ckpt: Path, bpe_path: Path, coarse_thresholds):
    if not sam3_ckpt.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {sam3_ckpt}")

    model_sam = build_sam3_image_model(
        bpe_path=str(bpe_path), checkpoint_path=str(sam3_ckpt), load_from_HF=False,
    )
    if torch.cuda.is_available():
        try:
            model_sam = model_sam.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[warn] CUDA OOM while moving SAM3 to GPU; fallback to CPU for cache build.")
                _safe_cuda_empty_cache()
            else:
                raise

    conf_min = min(coarse_thresholds)
    processor = Sam3Processor(model_sam, confidence_threshold=conf_min)
    return model_sam, processor


def build_coarse_cache(image_ids, target_classes, class_prompts, coarse_cache_dir: Path,
                        coarse_thresholds, sam3_ckpt: Path, bpe_path: Path,
                        force_rebuild: bool = False, gc_every: int = 64):
    """Runs SAM3 once, saves a binary (0/1) union mask per target class into .npz -- same format
    as the Cityscapes/VOC coarse cache. Each file: {image_id}.npz with keys = target_classes,
    value = (H,W) uint8 mask."""
    if not target_classes:
        print("[warn] target_classes is empty -- nothing to build. Create adjust_prompt_pc59.json first.")
        return []

    missing = [
        img_id for img_id in image_ids
        if force_rebuild or not (coarse_cache_dir / f"{img_id}.npz").exists()
    ]
    if not missing:
        print(f"Coarse cache complete ({len(image_ids)} images). Skipping.")
        return []

    print(f"Building coarse cache: {len(missing)}/{len(image_ids)} images...")
    model_sam, processor = build_sam3_processor(sam3_ckpt, bpe_path, coarse_thresholds)

    errors = []
    try:
        for step, img_id in enumerate(tqdm(missing, desc="SAM3 coarse cache (PC59, LLM)"), start=1):
            cache_path = coarse_cache_dir / f"{img_id}.npz"
            try:
                image = Image.open(PC59_IMAGES_ROOT / f"{img_id}.jpg").convert("RGB")
                state = processor.set_image(image)

                coarse_dict = {}
                for cls_name, prompts in zip(target_classes, class_prompts):
                    # Gather candidate (mask, score) pairs across ALL synonym prompts for this
                    # class, then union-at-threshold over the combined pool -- same semantics as
                    # the Cityscapes/VOC coarse cache.
                    all_masks, all_scores = [], []
                    for prompt in prompts:
                        processor.reset_all_prompts(state)
                        state = processor.set_text_prompt(state=state, prompt=prompt)
                        masks, scores = _extract_masks_scores(state)
                        all_masks.extend(masks)
                        all_scores.extend(scores)

                    union, _ = _union_at_threshold(
                        all_masks, all_scores, (image.height, image.width), coarse_thresholds
                    )
                    coarse_dict[cls_name] = union

                np.savez_compressed(cache_path, **coarse_dict)
            except Exception as e:
                errors.append((img_id, str(e)))

            if step % gc_every == 0:
                gc.collect()
                _safe_cuda_empty_cache()
    finally:
        try:
            del processor
        except Exception:
            pass
        del model_sam
        gc.collect()
        _safe_cuda_empty_cache()

    print(f"Done. Errors: {len(errors)}")
    for img_id, err in errors[:5]:
        print(f"  {img_id}: {err}")
    return errors


def audit_and_rebuild_cache(all_ids, target_classes, class_prompts, coarse_cache_dir: Path,
                             coarse_thresholds, sam3_ckpt: Path, bpe_path: Path):
    """Audit cache & rebuild only what is missing/corrupt (avoids reloading SAM3 repeatedly)."""
    req_keys = set(target_classes)

    def _cache_ok(img_id):
        p = coarse_cache_dir / f"{img_id}.npz"
        if not p.exists():
            return False, "missing_file"
        try:
            with np.load(p) as d:
                keys = set(d.files)
                miss = req_keys - keys
                if miss:
                    return False, f"missing_keys:{sorted(miss)}"
                for k in req_keys:
                    arr = d[k]
                    if arr.ndim != 2:
                        return False, f"bad_ndim:{k}:{arr.ndim}"
                return True, "ok"
        except Exception as e:
            return False, f"corrupt:{type(e).__name__}:{str(e)[:120]}"

    todo_ids, todo_reason = [], {}
    for img_id in tqdm(all_ids, desc="Audit coarse_cache_pc59"):
        ok, reason = _cache_ok(img_id)
        if not ok:
            todo_ids.append(img_id)
            todo_reason[img_id] = reason

    print(f"Need rebuild: {len(todo_ids)}/{len(all_ids)}")
    for img_id in todo_ids[:10]:
        print(f"  {img_id}: {todo_reason[img_id]}")

    if len(todo_ids) == 0:
        print("Cache already complete, no rebuild needed.")
    else:
        print("Rebuilding missing/corrupt cache in one pass (single SAM3 load)...")
        build_coarse_cache(todo_ids, target_classes, class_prompts, coarse_cache_dir,
                            coarse_thresholds, sam3_ckpt, bpe_path, force_rebuild=True)

    remain = [i for i in all_ids if not _cache_ok(i)[0]]
    print(f"Cache ready: {len(all_ids) - len(remain)}/{len(all_ids)}")
    print(f"Still missing/corrupt: {len(remain)}")
    print("Sample remain:", remain[:10])
    return remain


# ============================================================================
# Per-class GT pixel stats (for CE class weights downstream)
# ============================================================================

def scan_gt_class_stats(internal_train_ids, target_classes, gt_root: Path, coarse_cache_dir: Path):
    """Scan GT for per-class pixel stats on the internal TRAIN split (for CE class weights
    later). Scanned only on internal_train_ids (not internal_val, not the official val split) so
    the weights reflect exactly what the training loop will see. Unlike VOC/Cityscapes, PC59's
    raw mask has no genuine void/255 value -- every pixel counts as "valid"."""
    cls_pos_px = {c: 0 for c in target_classes}
    cls_neg_px = {c: 0 for c in target_classes}
    train_pos_ids = {c: set() for c in target_classes}
    missing_gt = []

    for img_id in tqdm(internal_train_ids, desc="Scanning GT for per-class pixel stats"):
        mask_path = get_gt_mask_path(img_id, gt_root)
        if mask_path is None:
            missing_gt.append(img_id)
            continue

        label_mask = load_pc59_mask(mask_path)
        total_valid = int(label_mask.size)  # every pixel is valid -- no void marker in PC59 raw masks

        for cls_name in target_classes:
            idx = GT_CLASS_TO_INDEX[cls_name]
            cls_mask = (label_mask == idx)
            p = int(cls_mask.sum())
            if p > 0:
                train_pos_ids[cls_name].add(img_id)
                cls_pos_px[cls_name] += p
                cls_neg_px[cls_name] += (total_valid - p)

    if missing_gt:
        print(f"[warn] {len(missing_gt)} images had no matching GT mask found (check PC59_GT_ROOT):")
        for m in missing_gt[:5]:
            print(f"  {m}")

    # Per-class weight: neg/pos ratio, clipped [1, 50] -- same convention as Cityscapes/VOC
    cls_pos_weights = {}
    for cls_name in target_classes:
        ratio = cls_neg_px[cls_name] / max(cls_pos_px[cls_name], 1)
        cls_pos_weights[cls_name] = float(np.clip(ratio, 1.0, 50.0))

    print(f"\nPer-class stats (internal train images, n={len(internal_train_ids)}):")
    for cls_name in target_classes:
        print(f"  {cls_name:12s} | pos_images={len(train_pos_ids[cls_name]):5d} "
              f"| pos_px={cls_pos_px[cls_name]:>13,} | neg_px={cls_neg_px[cls_name]:>15,} "
              f"| pos_weight={cls_pos_weights[cls_name]:.2f}")

    stats_path = coarse_cache_dir / "class_pixel_stats.json"
    class_stats = {
        cls_name: {
            "pos_images": len(train_pos_ids[cls_name]),
            "pos_px": cls_pos_px[cls_name],
            "neg_px": cls_neg_px[cls_name],
            "pos_weight": cls_pos_weights[cls_name],
        }
        for cls_name in target_classes
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(class_stats, f, ensure_ascii=False, indent=2)
    print(f"\nClass pixel stats saved to {stats_path}")
    return class_stats


# ============================================================================
# Packaging + sanity check
# ============================================================================

def zip_coarse_cache(coarse_cache_dir: Path, zip_path: Path):
    if not coarse_cache_dir.exists():
        print(f"[warn] Cache directory not found: {coarse_cache_dir}")
        return
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=str(coarse_cache_dir.parent),
        base_dir=coarse_cache_dir.name,
    )
    print(f"ZIP created: {zip_path}")
    print(f"Cache files: {len(list(coarse_cache_dir.glob('*.npz')))}")


def visualize_sample(internal_train_ids, target_classes, coarse_cache_dir: Path, out_path: Path):
    """Sanity check: visualize a couple of cached masks before committing to the full run."""
    if not internal_train_ids or not target_classes:
        print("[skip] no internal_train_ids or target_classes -- skipping sanity visualization")
        return
    img_id = internal_train_ids[0]
    cache_path = coarse_cache_dir / f"{img_id}.npz"
    if not cache_path.exists():
        print(f"[skip] no cache for {img_id} yet")
        return

    with np.load(cache_path) as d:
        n = len(target_classes)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
        axes = axes.flatten() if n > 1 else [axes]
        for ax, ch in zip(axes, target_classes):
            ax.imshow(d[ch], vmin=0, vmax=1, cmap="gray")
            ax.set_title(ch, fontsize=10)
            ax.axis("off")
        for ax in axes[len(target_classes):]:
            ax.axis("off")
        fig.suptitle(f"Coarse binary masks -- {img_id}", fontsize=13)
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
    print(f"Saved sanity-check visualization to {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Precompute SAM3 coarse masks for PC59 (LLM arm).")
    p.add_argument("--rebuild-cache", action="store_true",
                    help="Force rebuild every cache entry, even if it already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    setup_torch()
    set_seed(SEED)

    COARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    target_classes, class_prompts = load_prompt_ensemble(ADJUST_PROMPT_PATH)
    print(f"Target classes ({len(target_classes)}): {target_classes}")
    print(f"Prompt ensembles: {class_prompts}")
    print(f"Cache dir: {COARSE_CACHE_DIR}")
    print(f"SAM3 ckpt: {SAM3_CKPT}")

    sam3_dir = Path(sam3.__file__).parent
    bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    all_train_ids = discover_pc59_train_ids(TRAIN_FILE)
    print(f"Found {len(all_train_ids)} ids in {TRAIN_FILE}")
    if not all_train_ids:
        raise RuntimeError(f"No ids found in TRAIN_FILE={TRAIN_FILE}. Check the path.")

    internal_train_ids, internal_val_ids = split_internal_train_val(all_train_ids, COARSE_CACHE_DIR)

    # Step 1: build cache for every image missing one (or all, if --rebuild-cache)
    build_coarse_cache(
        all_train_ids, target_classes, class_prompts, COARSE_CACHE_DIR, COARSE_THRESHOLDS,
        SAM3_CKPT, bpe_path, force_rebuild=args.rebuild_cache,
    )

    # Step 2: audit the full cache and rebuild anything still missing/corrupt, in one more pass
    audit_and_rebuild_cache(
        all_train_ids, target_classes, class_prompts, COARSE_CACHE_DIR, COARSE_THRESHOLDS,
        SAM3_CKPT, bpe_path,
    )

    # Step 3: per-class GT pixel stats on the internal-train split (for downstream CE weights)
    scan_gt_class_stats(internal_train_ids, target_classes, PC59_GT_ROOT, COARSE_CACHE_DIR)

    # Step 4: package the final deliverable
    zip_coarse_cache(COARSE_CACHE_DIR, COARSE_CACHE_ZIP)

    # Step 5: quick visual sanity check (non-essential, best-effort)
    visualize_sample(internal_train_ids, target_classes, COARSE_CACHE_DIR,
                      COARSE_CACHE_DIR.parent / "sample_coarse_masks_pc59_llm.png")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed / 60:.1f} min")
    print(f"Deliverable zip: {COARSE_CACHE_ZIP}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] get_coarse_pc59_llm.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
