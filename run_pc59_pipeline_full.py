#!/usr/bin/env python3
"""
run_pc59_pipeline_full.py
============================
Full PC59 pipeline orchestrator -- BOTH arms (no-LLM and LLM), one command:

    [shared, run once]
    sam3_base_pc59_nollm.py            (baseline: full 59-class SAM3 eval, no-LLM prompts)
      -> select_refinement_classes.py  (fully automatic target-class selection)
      -> [bridge] write data_pc59/adjust_prompt_pc59.json (bare-name placeholder,
         skipped entirely if a real one from generate_adjust_prompt.py already exists)

    [no-LLM arm]
    get_coarse_pc59_nollm.py
      -> train_unetaspp_pc59_nollm.py        -> val_train_unetaspp_pc59_nollm.py
      -> train_unetasppdinov2_pc59_nollm.py  -> val_train_unetasppdinov2_pc59_nollm.py

    [LLM arm -- requires data_pc59/adjust_prompt_pc59.json to already contain REAL
     LLM-generated prompt ensembles, i.e. generate_adjust_prompt.py has already been run
     with your hand-built contexts_class_pc59.json. This script does NOT run class
     selection again for this arm -- it reuses the SAME target_classes.json.]
    sam3_baseline_pc59_llm.py          (LLM-arm baseline eval, for the paper's ablation
                                         table row -- no downstream script reads its output)
    get_coarse_pc59_llm.py
      -> train_unetaspp_pc59_llm.py          -> val_train_unetaspp_pc59_llm.py
      -> train_unetasppdinov2_pc59_llm.py    -> val_train_unetasppdinov2_pc59_llm.py

Every step calls the SAME already-verified .py file you already tested individually --
this script does not reimplement or modify any of their logic, only sequences them.

--------------------------------------------------------------------------------------------
IMPORTANT -- output-path collision fix (why this orchestrator exists, not just a loop):
SIX scripts all write to the SAME 4 shared filenames at PROJECT_ROOT:
per_class_metrics.csv, summary_metrics.csv, per_class_iou_bar_chart.png,
class_visualizations/ -- confirmed by inspecting all 11 files' CONFIG blocks:
    sam3_base_pc59_nollm.py, sam3_baseline_pc59_llm.py,
    val_train_unetaspp_pc59_nollm.py, val_train_unetaspp_pc59_llm.py,
    val_train_unetasppdinov2_pc59_nollm.py, val_train_unetasppdinov2_pc59_llm.py
Running them back-to-back without intervention would silently overwrite each other's
results, one after another, leaving only the LAST one's numbers behind. This script
archives those 4 paths into results_pc59/<step-name>_* immediately after each of those 6
steps, before the next one gets a chance to write there again. This is an
ORCHESTRATION-level fix -- none of the individual scripts are modified.

The bridge step exists because get_coarse/train/val (no-LLM) all read TARGET_CLASSES from
data_pc59/adjust_prompt_pc59.json's KEYS (discarding the prompt-ensemble VALUES, by
design), so both arms share one source of truth for which classes are targets.
select_refinement_classes.py outputs a different format (target_classes.json, a plain
list), so this script converts it into a minimal adjust_prompt_pc59.json (each selected
class mapped to [itself]) -- functionally IDENTICAL to what the no-LLM arm does
internally anyway, and ONLY written if the file doesn't already exist. If you already ran
generate_adjust_prompt.py (LLM arm), this script detects the real file and leaves it
untouched -- the bridge only ever fires as a no-LLM-only convenience for a from-scratch run.

Idempotent: every step is skipped if its expected output already exists, so re-running
this script after a partial run (or after a crash, or when the no-LLM arm was already
done in an earlier session) picks up where it left off. Delete the specific output
file/folder to force that one step to re-run.
--------------------------------------------------------------------------------------------

Usage:
    python run_pc59_pipeline_full.py
    python run_pc59_pipeline_full.py --skip-nollm         # LLM arm only (no-LLM already done)
    python run_pc59_pipeline_full.py --skip-llm           # no-LLM arm only
    python run_pc59_pipeline_full.py --force-train         # re-run all training even if checkpoints exist
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results_pc59"

# Shared-filename outputs that collide across all 6 producing scripts (see module
# docstring). Archived under this exact set of names after each producing step.
SHARED_OUTPUTS = [
    ("per_class_metrics.csv", "file"),
    ("summary_metrics.csv", "file"),
    ("per_class_iou_bar_chart.png", "file"),
    ("class_visualizations", "dir"),
]

TARGET_CLASSES_JSON = PROJECT_ROOT / "target_classes.json"
ADJUST_PROMPT_PATH = PROJECT_ROOT / "data_pc59" / "adjust_prompt_pc59.json"

BASELINE_CSV_ARCHIVED = RESULTS_DIR / "00_baseline_nollm_per_class_metrics.csv"

COARSE_CACHE_ZIP_NOLLM = PROJECT_ROOT / "coarse_cache_pc59_nollm.zip"
COARSE_CACHE_ZIP_LLM = PROJECT_ROOT / "coarse_cache_pc59.zip"

ASPP_CKPT_NOLLM = PROJECT_ROOT / "weights_unetaspp_pc59_nollm_v1" / "unetaspp_pc59_nollm_v1_best.pth"
DINOV2_CKPT_NOLLM = PROJECT_ROOT / "weights_aspp_pc59_nollm_v1" / "unet_aspp_pc59_nollm_v1_best.pth"
ASPP_CKPT_LLM = PROJECT_ROOT / "weights_unetaspp_pc59_llm_v1" / "unetaspp_pc59_llm_v1_best.pth"
DINOV2_CKPT_LLM = PROJECT_ROOT / "weights_aspp_pc59_llm_v1" / "unet_aspp_pc59_llm_v1_best.pth"

BASELINE_LLM_ARCHIVED = RESULTS_DIR / "03_baseline_llm_per_class_metrics.csv"


def log(msg: str) -> None:
    print(f"\n{'='*78}\n{msg}\n{'='*78}", flush=True)


def run_script(script_name: str, args=None) -> None:
    """Run one of the already-verified pipeline .py files as a subprocess, using the SAME
    Python interpreter (venv) this orchestrator is running under, from PROJECT_ROOT (so
    each script's own Path(__file__).resolve().parent resolves correctly and every
    hardcoded relative path inside it lines up)."""
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Expected pipeline script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + (args or [])
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    print(f"[done in {time.time() - t0:.1f}s] {script_name}", flush=True)


def archive_shared_outputs(tag: str) -> None:
    """Move the 4 shared-filename outputs into results_pc59/<tag>_* right after a step
    that writes them, before the next such step can overwrite them."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, kind in SHARED_OUTPUTS:
        src = PROJECT_ROOT / name
        if not src.exists():
            print(f"[warn] expected output not found, nothing to archive: {src}", file=sys.stderr)
            continue
        dst = RESULTS_DIR / f"{tag}_{name}"
        if dst.exists():
            if kind == "dir":
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        print(f"  archived: {name} -> results_pc59/{tag}_{name}")


def write_bridge_adjust_prompt(target_classes: list) -> None:
    """target_classes.json (a list) -> data_pc59/adjust_prompt_pc59.json (the dict format
    get_coarse/train/val (no-LLM) expect), each class mapped to [itself] as the only
    prompt -- identical to what the no-LLM arm computes internally regardless, see module
    docstring. Never overwrites a richer adjust_prompt_pc59.json that generate_adjust_prompt.py
    may have already produced for the LLM arm -- only fills it in if genuinely missing."""
    if ADJUST_PROMPT_PATH.exists():
        with open(ADJUST_PROMPT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        existing_classes = set(existing.keys())
        target_set = set(target_classes)

        looks_like_placeholder = all(v == [k] for k, v in existing.items())
        print(f"[skip] {ADJUST_PROMPT_PATH.relative_to(PROJECT_ROOT)} already exists"
              + (" (looks like a bare-name PLACEHOLDER, not real LLM prompts -- fine for "
                 "the no-LLM arm, but the LLM arm needs generate_adjust_prompt.py run first)"
                 if looks_like_placeholder else " (has real, non-trivial prompt values)")
              + " -- not overwriting.")

        if existing_classes != target_set:
            print(f"[warn] existing adjust_prompt_pc59.json's classes do NOT match "
                  f"target_classes.json's selection. In file only: "
                  f"{sorted(existing_classes - target_set)}. In selection only: "
                  f"{sorted(target_set - existing_classes)}. Resolve this manually before "
                  f"continuing -- get_coarse/train will use whatever is in the file.",
                  file=sys.stderr)
        return

    ADJUST_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    placeholder = {cls: [cls] for cls in target_classes}
    with open(ADJUST_PROMPT_PATH, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, indent=2, ensure_ascii=False)
    print(f"Wrote bridge file: {ADJUST_PROMPT_PATH}  ({len(target_classes)} classes, bare-name "
          f"placeholder -- the no-LLM arm discards prompt VALUES anyway, only reads the keys. "
          f"The LLM arm needs a REAL adjust_prompt_pc59.json from generate_adjust_prompt.py "
          f"before running --skip-nollm later.)")


def run_shared_steps(args) -> list:
    """Baseline (no-LLM) + automatic class selection + bridge file. Returns target_classes."""
    log("STEP 1: SAM3 baseline (full 59-class eval, no-LLM)")
    baseline_csv_standalone = PROJECT_ROOT / "per_class_metrics.csv"
    if BASELINE_CSV_ARCHIVED.exists() and not args.force_baseline:
        print(f"[skip] {BASELINE_CSV_ARCHIVED.relative_to(PROJECT_ROOT)} already exists. Use --force-baseline to re-run.")
    elif baseline_csv_standalone.exists() and not args.force_baseline:
        # Baseline was already run manually/standalone (outside this orchestrator, e.g. in
        # an earlier session) and never archived -- archive it now instead of expensively
        # re-running SAM3 on all ~5000 images for nothing.
        print(f"[skip] {baseline_csv_standalone.name} already exists at PROJECT_ROOT (from an "
              f"earlier standalone run) -- archiving it instead of re-running. Use "
              f"--force-baseline to force a fresh baseline eval.")
        archive_shared_outputs("00_baseline_nollm")
    else:
        run_script("sam3_base_pc59_nollm.py")
        # NOTE: do not copy this back to PROJECT_ROOT after archiving -- 5 other scripts
        # later in the run write the SAME transient filename and would just delete it
        # again with nothing to restore it, breaking this skip-check on a future run.
        # select_refinement_classes.py reads directly from the archived path instead.
        archive_shared_outputs("00_baseline_nollm")

    log("STEP 2: Automatic target-class selection (BIC-gated auto: GMM or cumulative budget)")
    if TARGET_CLASSES_JSON.exists() and not args.force_select:
        print(f"[skip] {TARGET_CLASSES_JSON.name} already exists. Use --force-select to re-run.")
    else:
        run_script("select_refinement_classes.py", [
            "--input", str(BASELINE_CSV_ARCHIVED),
            "--output", str(TARGET_CLASSES_JSON),
            "--plot", str(RESULTS_DIR / "01_refinement_priority_curve.png"),
            "--alpha", str(args.alpha), "--beta", str(args.beta),
        ])

    with open(TARGET_CLASSES_JSON, encoding="utf-8") as f:
        selection = json.load(f)
    target_classes = selection["target_classes"]
    print(f"\nTarget classes ({len(target_classes)}): {target_classes}")

    write_bridge_adjust_prompt(target_classes)
    return target_classes


def run_nollm_arm(args) -> None:
    log("STEP 3: SAM3 coarse mask cache (get_coarse, no-LLM)")
    if COARSE_CACHE_ZIP_NOLLM.exists() and not args.force_coarse:
        print(f"[skip] {COARSE_CACHE_ZIP_NOLLM.name} already exists. Use --force-coarse to re-run.")
    else:
        run_script("get_coarse_pc59_nollm.py")

    log("STEP 4: Train UNet+ASPP (no-LLM)")
    if ASPP_CKPT_NOLLM.exists() and not args.force_train:
        print(f"[skip] {ASPP_CKPT_NOLLM.name} already exists. Use --force-train to re-run.")
    else:
        run_script("train_unetaspp_pc59_nollm.py")

    log("STEP 5: Train UNet+ASPP+DINOv2 (no-LLM)")
    if DINOV2_CKPT_NOLLM.exists() and not args.force_train:
        print(f"[skip] {DINOV2_CKPT_NOLLM.name} already exists. Use --force-train to re-run.")
    else:
        run_script("train_unetasppdinov2_pc59_nollm.py")

    log("STEP 6: Val -- hybrid eval, no-LLM, both architectures")
    run_script("val_train_unetaspp_pc59_nollm.py")
    archive_shared_outputs("01_val_unetaspp_nollm")

    run_script("val_train_unetasppdinov2_pc59_nollm.py")
    archive_shared_outputs("02_val_unetasppdinov2_nollm")


def run_llm_arm(args) -> None:
    if not ADJUST_PROMPT_PATH.exists():
        raise RuntimeError(
            f"{ADJUST_PROMPT_PATH} does not exist -- cannot run the LLM arm. Build "
            f"contexts_class_pc59.json and run generate_adjust_prompt.py first."
        )
    with open(ADJUST_PROMPT_PATH, encoding="utf-8") as f:
        adjust_prompt = json.load(f)
    if all(v == [k] for k, v in adjust_prompt.items()):
        print(f"[warn] {ADJUST_PROMPT_PATH.relative_to(PROJECT_ROOT)} looks like the bare-name "
              f"PLACEHOLDER (every value is just [class_name]), not real LLM-generated prompts. "
              f"Run generate_adjust_prompt.py first if you haven't yet -- continuing anyway, "
              f"but the LLM arm's results will be identical to the no-LLM arm's if this is "
              f"still the placeholder.", file=sys.stderr)

    log("STEP 7: SAM3 baseline (full 59-class eval, LLM prompts)")
    baseline_llm_standalone = PROJECT_ROOT / "per_class_metrics.csv"
    if BASELINE_LLM_ARCHIVED.exists() and not args.force_baseline:
        print(f"[skip] {BASELINE_LLM_ARCHIVED.relative_to(PROJECT_ROOT)} already exists. Use --force-baseline to re-run.")
    elif baseline_llm_standalone.exists() and not args.force_baseline:
        print(f"[skip] {baseline_llm_standalone.name} already exists at PROJECT_ROOT (from an "
              f"earlier standalone run) -- archiving it instead of re-running. Use "
              f"--force-baseline to force a fresh baseline eval.")
        archive_shared_outputs("03_baseline_llm")
    else:
        run_script("sam3_baseline_pc59_llm.py")
        archive_shared_outputs("03_baseline_llm")

    log("STEP 8: SAM3 coarse mask cache (get_coarse, LLM)")
    if COARSE_CACHE_ZIP_LLM.exists() and not args.force_coarse:
        print(f"[skip] {COARSE_CACHE_ZIP_LLM.name} already exists. Use --force-coarse to re-run.")
    else:
        run_script("get_coarse_pc59_llm.py")

    log("STEP 9: Train UNet+ASPP (LLM)")
    if ASPP_CKPT_LLM.exists() and not args.force_train:
        print(f"[skip] {ASPP_CKPT_LLM.name} already exists. Use --force-train to re-run.")
    else:
        run_script("train_unetaspp_pc59_llm.py")

    log("STEP 10: Train UNet+ASPP+DINOv2 (LLM)")
    if DINOV2_CKPT_LLM.exists() and not args.force_train:
        print(f"[skip] {DINOV2_CKPT_LLM.name} already exists. Use --force-train to re-run.")
    else:
        run_script("train_unetasppdinov2_pc59_llm.py")

    log("STEP 11: Val -- hybrid eval, LLM, both architectures")
    run_script("val_train_unetaspp_pc59_llm.py")
    archive_shared_outputs("04_val_unetaspp_llm")

    run_script("val_train_unetasppdinov2_pc59_llm.py")
    archive_shared_outputs("05_val_unetasppdinov2_llm")


def main():
    parser = argparse.ArgumentParser(description="Full PC59 pipeline orchestrator (no-LLM + LLM).")
    parser.add_argument("--skip-nollm", action="store_true", help="skip the no-LLM arm entirely (assumes it's already done)")
    parser.add_argument("--skip-llm", action="store_true", help="skip the LLM arm entirely")
    parser.add_argument("--force-baseline", action="store_true", help="re-run baseline eval(s) even if already archived")
    parser.add_argument("--force-select", action="store_true", help="re-run class selection even if target_classes.json exists")
    parser.add_argument("--force-coarse", action="store_true", help="re-run get_coarse even if the cache zip exists")
    parser.add_argument("--force-train", action="store_true", help="re-run training even if a best checkpoint exists")
    parser.add_argument("--alpha", type=float, default=0.5, help="passed through to select_refinement_classes.py")
    parser.add_argument("--beta", type=float, default=0.5, help="passed through to select_refinement_classes.py")
    args = parser.parse_args()

    pipeline_t0 = time.time()

    run_shared_steps(args)

    if not args.skip_nollm:
        run_nollm_arm(args)
    else:
        print("\n--skip-nollm set -- no-LLM arm skipped.")

    if not args.skip_llm:
        run_llm_arm(args)
    else:
        print("\n--skip-llm set -- LLM arm skipped.")

    elapsed = time.time() - pipeline_t0
    log(f"PIPELINE COMPLETE in {elapsed/60:.1f} min")
    print(f"Target classes         : {TARGET_CLASSES_JSON}, {RESULTS_DIR}/01_refinement_priority_curve.png")
    if not args.skip_nollm:
        print(f"Baseline (no-LLM)      : {RESULTS_DIR}/00_baseline_nollm_*")
        print(f"UNet+ASPP (no-LLM)     : {RESULTS_DIR}/01_val_unetaspp_nollm_*")
        print(f"+DINOv2 (no-LLM)       : {RESULTS_DIR}/02_val_unetasppdinov2_nollm_*")
    if not args.skip_llm:
        print(f"Baseline (LLM)         : {RESULTS_DIR}/03_baseline_llm_*")
        print(f"UNet+ASPP (LLM)        : {RESULTS_DIR}/04_val_unetaspp_llm_*")
        print(f"+DINOv2 (LLM)          : {RESULTS_DIR}/05_val_unetasppdinov2_llm_*")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] Pipeline stopped -- '{Path(e.cmd[1]).name}' exited with code {e.returncode}. "
              f"Fix the error above and re-run this orchestrator (already-completed steps will be skipped).",
              file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("\n[FATAL] run_pc59_pipeline_full.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)