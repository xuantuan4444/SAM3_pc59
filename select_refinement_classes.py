#!/usr/bin/env python3
"""
select_refinement_classes.py
==============================
Automatic target-class selection for GenMask-SAM3 -- replaces the manual IoU-threshold
criterion (C_w = {c | IoU_c < tau}) with a Refinement Priority Score + automatic
knee/elbow detection, following the advisor's proposal (sam3_auto_idea.docx).

    R_c = alpha * D_c + beta * E_c

    D_c = normalized performance deficit  = (1 - IoU_c) / max_c(1 - IoU_c)
    E_c = normalized error contribution   = (FP_c + FN_c) / sum_c(FP_c + FN_c)

    C_w = KneeSelect({R_c}_{c=1..C})   via the Kneedle algorithm (Satopaa et al. 2011)

FP_c/FN_c are NOT read from a separate column -- they are algebraically recovered from
the 3 columns sam3_base_pc59_nollm.py's compute_metrics() already saves (IoU, GT Pixels,
Pred Pixels), so this module works unmodified against the EXISTING baseline CSV format;
no change needed to any already-verified baseline script.

Derivation: IoU = TP/(TP+FP+FN), GT Pixels = TP+FN, Pred Pixels = TP+FP
    =>  U  = (GT Pixels + Pred Pixels) / (1 + IoU)      # = TP+FP+FN
        TP = IoU * U
        FP = Pred Pixels - TP
        FN = GT Pixels - TP

--------------------------------------------------------------------------------------------
IMPORTANT direction note (easy to get backwards -- documented here deliberately):
The advisor's own worked example sorts raw IoU ASCENDING and keeps everything BEFORE the
knee (classes[:knee_idx]), because for raw IoU, low values = bad = needs refinement, and
the curve rises from bad to good.

For the PRIORITY score used here, the semantics are inverted: HIGH priority = needs
refinement. Sorted ascending, the curve still rises left-to-right (same S-shape), but the
"needs refinement" classes are now the ones AFTER/AT the knee (the high tail), not before
it. This module selects the tail -- do not copy the advisor's classes[:knee_idx] slicing
verbatim onto a priority-sorted array, it would silently select the WRONG classes.
--------------------------------------------------------------------------------------------

Input : per_class_metrics.csv -- produced by sam3_base_pc59_nollm.py (or any baseline
        script following the same compute_metrics() format: columns Class, IoU, Dice,
        GT Pixels, Pred Pixels)
Output: target_classes.json          -- selected classes + full priority ranking
        refinement_priority_curve.png -- diagnostic plot, ALWAYS inspect this, don't
                                          trust the knee index blindly (same caution the
                                          advisor gave for the original elbow method)

Setup:
    pip install kneed scikit-learn pandas numpy matplotlib

Usage:
    python select_refinement_classes.py --input per_class_metrics.csv --method knee
    python select_refinement_classes.py --input per_class_metrics.csv --method gmm

Which method to use: Kneedle (--method knee, the default) finds a genuine geometric bend
and works well when the priority curve has 2 visually-separated clusters (typically true
for smaller class counts, e.g. Cityscapes 19 classes, Pascal VOC 20 classes). For larger,
more continuous label spaces (e.g. PC59's 59 classes) the priority curve is often a smooth
monotonic continuum with no true bend -- in that case Kneedle will fixate on whatever the
single largest local step happens to be, and tuning --sensitivity will NOT change the
result (there is only one candidate point for it to find, not several to choose among).
Use --method gmm instead: it fits a 2-component Gaussian Mixture on the priority
distribution and selects the high-priority component, which works regardless of curve
shape. ALWAYS inspect the saved plot before trusting either method's output.
--------------------------------------------------------------------------------------------
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe, no X server needed on a remote server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# kneed (--method knee) and scikit-learn (--method gmm) are imported lazily inside their
# respective select_via_* functions, so only the library needed for the chosen --method
# has to be installed.


# ============================================================================
# FP/FN recovery (no changes needed to the baseline script that produced the CSV)
# ============================================================================

def recover_tp_fp_fn(iou: float, gt_pixels: float, pred_pixels: float):
    """Algebraically recover (TP, FP, FN) from IoU + GT Pixels + Pred Pixels.
    See module docstring for the derivation."""
    denom = 1.0 + iou
    union = (gt_pixels + pred_pixels) / denom if denom > 0 else 0.0
    tp = iou * union
    fp = max(pred_pixels - tp, 0.0)
    fn = max(gt_pixels - tp, 0.0)
    return tp, fp, fn


# ============================================================================
# Priority score
# ============================================================================

def compute_priority(class_df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    df = class_df.copy()

    recovered = df.apply(
        lambda row: recover_tp_fp_fn(row["IoU"], row["GT Pixels"], row["Pred Pixels"]),
        axis=1, result_type="expand",
    )
    recovered.columns = ["TP", "FP", "FN"]
    df = pd.concat([df, recovered], axis=1)
    df["Error"] = df["FP"] + df["FN"]

    deficit = 1.0 - df["IoU"]
    max_deficit = deficit.max()
    df["Deficit_norm"] = deficit / max_deficit if max_deficit > 0 else 0.0

    total_error = df["Error"].sum()
    df["Error_contribution"] = df["Error"] / total_error if total_error > 0 else 0.0

    df["Priority"] = alpha * df["Deficit_norm"] + beta * df["Error_contribution"]

    return df.sort_values("Priority", ascending=False).reset_index(drop=True)


# ============================================================================
# Knee/Elbow selection (Kneedle)
# ============================================================================

def select_via_knee(priority_df: pd.DataFrame, sensitivity: float = 1.0):
    """Sort ascending by Priority (Kneedle expects this orientation), find the knee,
    and select the HIGH-priority tail (at/after the knee) -- see the direction note in
    the module docstring for why this is the tail, not the head.

    sensitivity (Kneedle's S parameter): LOWER = more sensitive, detects gentler bends
    (use this if the curve is a smooth/gradual continuum rather than 2 sharply-separated
    clusters -- default S=1.0 can end up flagging only the single most extreme outlier
    on such curves, which is too conservative). Try 0.5, 0.3, or 0 if the default
    under-selects."""
    try:
        from kneed import KneeLocator
    except ImportError:
        print("[FATAL] --method knee requires: pip install kneed", file=sys.stderr)
        raise

    sorted_df = priority_df.sort_values("Priority", ascending=True).reset_index(drop=True)
    x = np.arange(1, len(sorted_df) + 1)
    y = sorted_df["Priority"].values

    knee = KneeLocator(x, y, S=sensitivity, curve="convex", direction="increasing")
    knee_idx = knee.knee  # 1-indexed position in the ascending-sorted array, or None
    knee_idx = int(knee_idx) if knee_idx is not None else None  # numpy.int64 -> native int (JSON-safe)

    if knee_idx is None:
        print("[warn] No clear knee point found by Kneedle -- returning an EMPTY selection.",
              file=sys.stderr)
        print("[warn] Inspect refinement_priority_curve.png. If the curve genuinely has no "
              "bend (near-linear or too few classes), fall back to a manual threshold on "
              "the Priority column instead of trusting automatic selection here.",
              file=sys.stderr)
        return [], sorted_df, None

    selected = sorted_df.iloc[knee_idx - 1:]["Class"].tolist()  # tail = high priority = needs refinement

    # Sanity check: warn loudly if the selection looks suspiciously small relative to how
    # many classes are clearly underperforming (IoU < 0.5) -- catches the exact failure
    # mode observed on PC59 (S=1.0 flagging only 2/59 despite dozens of IoU<0.5 classes).
    n_clearly_weak = int((sorted_df.get("IoU", pd.Series(dtype=float)) < 0.5).sum()) if "IoU" in sorted_df else None
    if n_clearly_weak is not None and len(selected) < 0.5 * n_clearly_weak:
        print(f"[warn] Selected only {len(selected)} classes, but {n_clearly_weak} classes have "
              f"IoU < 0.5. This usually means the priority curve is a smooth continuum (common "
              f"with 50+ classes) rather than 2 sharply-separated clusters, and Kneedle's default "
              f"sensitivity is too conservative for it. Try a lower --sensitivity (e.g. 0.5, 0.3, "
              f"or 0) and compare.", file=sys.stderr)

    return selected, sorted_df, knee_idx


# ============================================================================
# GMM-based selection -- more robust than Kneedle when the priority curve is a smooth,
# continuous distribution (common once the class count grows past ~30-40) rather than 2
# sharply-separated clusters. Same core idea (automatic, no manual threshold to justify)
# but relies on statistical separability of the 1-D priority distribution instead of a
# geometric bend, so it does not require a visible "elbow" to exist.
# ============================================================================

def select_via_gmm(priority_df: pd.DataFrame, n_components: int = 2, random_state: int = 42):
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        print("[FATAL] --method gmm requires: pip install scikit-learn", file=sys.stderr)
        raise

    priorities = priority_df["Priority"].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state, n_init=10)
    labels = gmm.fit_predict(priorities)

    means = gmm.means_.flatten()
    high_component = int(np.argmax(means))  # the cluster with the HIGHEST mean priority = "needs refinement"

    selected_mask = labels == high_component
    result_df = priority_df.copy()
    result_df["GMM_component"] = labels
    result_df["GMM_selected"] = selected_mask

    selected = priority_df.loc[selected_mask, "Class"].tolist()

    print(f"\nGMM component means: {sorted(means.tolist())}")
    print(f"Selected component: {high_component} (mean priority = {means[high_component]:.4f})")

    return selected, result_df, high_component


# ============================================================================
# BIC-based statistical validation -- always computed, regardless of --method, so the
# selection's statistical grounding is visible for every run on every dataset. Follows
# the Kass & Raftery (1995) evidence categories for a BIC difference:
#   < 2   : weak (no real support for 2 clusters over 1)
#   2-6   : positive
#   6-10  : strong
#   > 10  : very strong
# ============================================================================

def compute_bic_evidence(priority_df: pd.DataFrame, random_state: int = 42):
    from sklearn.mixture import GaussianMixture

    priorities = priority_df["Priority"].values.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=random_state, n_init=10).fit(priorities)
    gmm2 = GaussianMixture(n_components=2, random_state=random_state, n_init=10).fit(priorities)
    bic1, bic2 = gmm1.bic(priorities), gmm2.bic(priorities)
    delta = bic1 - bic2  # positive = 2 components fit better

    if delta < 2:
        strength = "weak"
    elif delta < 6:
        strength = "positive"
    elif delta < 10:
        strength = "strong"
    else:
        strength = "very strong"

    return {"bic_1_component": float(bic1), "bic_2_component": float(bic2),
            "delta_bic": float(delta), "evidence_strength": strength}


# ============================================================================
# Cumulative error-budget selection -- the fully-automatic fallback for when BIC
# evidence for a genuine 2-cluster split is weak (i.e. the priority distribution looks
# like a smooth continuum, not 2 separated populations, so imposing ANY hard 2-way split
# -- Kneedle or GMM alike -- would not be statistically grounded). Does NOT assume
# bimodality: sorts classes by Priority descending and keeps adding classes until their
# cumulative share of total error reaches a fixed, pre-declared budget. Still uses the
# full R_c signal (IoU deficit + error contribution), not raw IoU alone.
# ============================================================================

def select_via_cumulative_budget(priority_df: pd.DataFrame, budget: float = 0.80):
    sorted_df = priority_df.sort_values("Priority", ascending=False).reset_index(drop=True)
    cum_error = sorted_df["Error_contribution"].cumsum()
    sorted_df["Cumulative_error_share"] = cum_error

    # first index where cumulative share reaches the budget (inclusive)
    reach_idx = int((cum_error >= budget).idxmax()) if (cum_error >= budget).any() else len(sorted_df) - 1
    selected = sorted_df.iloc[:reach_idx + 1]["Class"].tolist()

    return selected, sorted_df, reach_idx + 1


# ============================================================================
# Fully automatic dispatcher -- the actual "method" for the paper: compute BIC evidence,
# branch to GMM (strong evidence) or cumulative budget (weak evidence). No human
# decision point for ANY dataset -- same script, same rule, applied uniformly.
# ============================================================================

def select_auto(priority_df: pd.DataFrame, strong_threshold: float = 2.0, budget: float = 0.80):
    bic_info = compute_bic_evidence(priority_df)
    print(f"\nBIC evidence for a genuine 2-cluster split: delta={bic_info['delta_bic']:.2f} "
          f"({bic_info['evidence_strength']})")

    if bic_info["delta_bic"] >= strong_threshold:
        print(f"  -> delta >= {strong_threshold}: sufficient evidence for a bimodal split. "
              f"Using GMM 2-component selection.")
        selected, result_df, high_component = select_via_gmm(priority_df)
        method_used = "gmm"
        extra = {"gmm_high_component": high_component}
    else:
        print(f"  -> delta < {strong_threshold}: NOT enough evidence for a bimodal split "
              f"(priority distribution looks like a continuum, not 2 separated populations). "
              f"Falling back to cumulative error-budget selection (budget={budget:.0%}) -- "
              f"still fully automatic, no bimodality assumed.")
        selected, result_df, reach_idx = select_via_cumulative_budget(priority_df, budget=budget)
        method_used = "cumulative"
        extra = {"cumulative_budget": budget, "cumulative_reach_index": reach_idx}

    return selected, result_df, method_used, bic_info, extra


# ============================================================================
# Diagnostic plot -- always generated, always inspect before trusting the selection
# ============================================================================

def plot_curve(sorted_df: pd.DataFrame, knee_idx, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_df) * 0.22), 6))
    x = np.arange(1, len(sorted_df) + 1)
    colors = ["#d9534f" if knee_idx is not None and i >= knee_idx else "#2ca02c" for i in x]

    ax.bar(x, sorted_df["Priority"], color=colors, alpha=0.8)
    if knee_idx is not None:
        ax.axvline(knee_idx - 0.5, linestyle="--", color="black", linewidth=1.5,
                    label=f"Knee (selection starts at rank {knee_idx})")
    ax.set_xlabel("Classes ranked by Refinement Priority (ascending)")
    ax.set_ylabel("Refinement Priority Score")
    ax.set_title("Refinement Priority Curve -- Automatic Knee Detection\n"
                  "(red = selected for refinement, green = kept as-is)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["Class"], rotation=65, ha="right", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_gmm(result_df: pd.DataFrame, out_path: Path):
    sorted_df = result_df.sort_values("Priority", ascending=True).reset_index(drop=True)
    x = np.arange(1, len(sorted_df) + 1)
    colors = ["#d9534f" if s else "#2ca02c" for s in sorted_df["GMM_selected"]]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_df) * 0.22), 6))
    ax.bar(x, sorted_df["Priority"], color=colors, alpha=0.8)
    ax.set_xlabel("Classes ranked by Refinement Priority (ascending)")
    ax.set_ylabel("Refinement Priority Score")
    ax.set_title("Refinement Priority -- GMM 2-component Selection\n"
                  "(red = high-priority component / selected, green = low-priority component)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df["Class"], rotation=65, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_cumulative(sorted_df: pd.DataFrame, reach_idx: int, budget: float, out_path: Path):
    """sorted_df here is DESCENDING by Priority (cumulative budget is naturally read that
    way -- rank 1 = highest priority)."""
    x = np.arange(1, len(sorted_df) + 1)
    colors = ["#d9534f" if i <= reach_idx else "#2ca02c" for i in x]

    fig, ax1 = plt.subplots(figsize=(max(10, len(sorted_df) * 0.22), 6))
    ax1.bar(x, sorted_df["Priority"], color=colors, alpha=0.8)
    ax1.set_xlabel("Classes ranked by Refinement Priority (descending)")
    ax1.set_ylabel("Refinement Priority Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_df["Class"], rotation=65, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, sorted_df["Cumulative_error_share"], color="black", marker="o", markersize=3,
              linewidth=1.5, label="Cumulative error share")
    ax2.axhline(budget, linestyle="--", color="blue", linewidth=1,
                 label=f"Budget = {budget:.0%}")
    ax2.axvline(reach_idx + 0.5, linestyle="--", color="black", linewidth=1)
    ax2.set_ylabel("Cumulative share of total error")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right")

    ax1.set_title(f"Refinement Priority -- Cumulative Error-Budget Selection\n"
                   f"(red = selected, {reach_idx}/{len(sorted_df)} classes reach {budget:.0%} of total error)")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automatic target-class selection via Refinement Priority + BIC-gated Knee/GMM/Cumulative."
    )
    parser.add_argument("--input", required=True,
                         help="per_class_metrics.csv from a baseline eval script")
    parser.add_argument("--output", default="target_classes.json")
    parser.add_argument("--plot", default="refinement_priority_curve.png")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for performance deficit")
    parser.add_argument("--beta", type=float, default=0.5, help="weight for error contribution")
    parser.add_argument("--method", choices=["auto", "knee", "gmm", "cumulative"], default="auto",
                         help="'auto' (default, use this for the actual pipeline/paper numbers): "
                              "fully automatic, no human decision point -- computes BIC evidence "
                              "for a genuine 2-cluster split and branches to GMM (strong evidence) "
                              "or cumulative error-budget (weak evidence) accordingly, the SAME "
                              "rule applied uniformly to every dataset. 'knee'/'gmm'/'cumulative' "
                              "force one specific method regardless of BIC -- for comparison/debugging only.")
    parser.add_argument("--bic-threshold", type=float, default=2.0,
                         help="[auto method only] minimum delta-BIC to trust a 2-cluster split "
                              "(2.0 = Kass & Raftery's 'positive evidence' threshold).")
    parser.add_argument("--budget", type=float, default=0.80,
                         help="[auto's cumulative fallback, or --method cumulative] cumulative "
                              "share of total error the selected classes must account for.")
    parser.add_argument("--sensitivity", type=float, default=1.0,
                         help="[--method knee only] Kneedle's S parameter -- lower = more "
                              "sensitive. Note: if the curve has only ONE candidate bend point "
                              "(common for smooth/monotonic curves), varying S will NOT change "
                              "the result.")
    parser.add_argument("--gmm-components", type=int, default=2,
                         help="[--method gmm only] number of mixture components")
    args = parser.parse_args()

    class_df = pd.read_csv(args.input)
    required_cols = {"Class", "IoU", "GT Pixels", "Pred Pixels"}
    missing = required_cols - set(class_df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}. "
                          f"Found columns: {list(class_df.columns)}")

    priority_df = compute_priority(class_df, alpha=args.alpha, beta=args.beta)

    # BIC diagnostic -- ALWAYS computed and printed, regardless of --method, so the
    # statistical grounding of the selection is visible on every run, every dataset.
    bic_info = compute_bic_evidence(priority_df)

    print("=" * 70)
    print(f"Refinement Priority ranking (alpha={args.alpha}, beta={args.beta}, method={args.method})")
    print("=" * 70)
    for _, row in priority_df.iterrows():
        print(f"  {row['Class']:15s}  IoU={row['IoU']:.4f}  "
              f"Deficit={row['Deficit_norm']:.4f}  ErrShare={row['Error_contribution']:.4f}  "
              f"Priority={row['Priority']:.4f}")

    print(f"\nBIC evidence for a genuine 2-cluster split: "
          f"BIC(k=1)={bic_info['bic_1_component']:.2f}  BIC(k=2)={bic_info['bic_2_component']:.2f}  "
          f"delta={bic_info['delta_bic']:.2f}  ({bic_info['evidence_strength']})")

    if args.method == "auto":
        selected, result_df, method_used, _, extra = select_auto(
            priority_df, strong_threshold=args.bic_threshold, budget=args.budget)
        if method_used == "gmm":
            plot_gmm(result_df, Path(args.plot))
        else:
            plot_cumulative(result_df, extra["cumulative_reach_index"] - 1, args.budget, Path(args.plot))
        method_meta = {"method_used": method_used, "bic_threshold": args.bic_threshold, **extra}

    elif args.method == "knee":
        selected, sorted_df, knee_idx = select_via_knee(priority_df, sensitivity=args.sensitivity)
        plot_curve(sorted_df, knee_idx, Path(args.plot))
        method_meta = {"sensitivity": args.sensitivity, "knee_index_ascending": knee_idx}

    elif args.method == "gmm":
        selected, result_df, high_component = select_via_gmm(priority_df, n_components=args.gmm_components)
        plot_gmm(result_df, Path(args.plot))
        method_meta = {"gmm_components": args.gmm_components, "gmm_high_component": high_component}

    else:  # cumulative
        selected, sorted_df, reach_idx = select_via_cumulative_budget(priority_df, budget=args.budget)
        plot_cumulative(sorted_df, reach_idx - 1, args.budget, Path(args.plot))
        method_meta = {"budget": args.budget, "cumulative_reach_index": reach_idx}

    print(f"\nSelected {len(selected)}/{len(priority_df)} target classes:")
    for c in selected:
        print(f"  - {c}")

    output = {
        "method": args.method,
        "alpha": args.alpha,
        "beta": args.beta,
        "bic": bic_info,
        **method_meta,
        "num_total_classes": len(priority_df),
        "num_selected": len(selected),
        "target_classes": selected,
        "full_ranking": [
            {
                "class": row["Class"],
                "iou": round(float(row["IoU"]), 6),
                "deficit_norm": round(float(row["Deficit_norm"]), 6),
                "error_contribution": round(float(row["Error_contribution"]), 6),
                "priority": round(float(row["Priority"]), 6),
                "selected": bool(row["Class"] in selected),
            }
            for _, row in priority_df.iterrows()
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] select_refinement_classes.py failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)