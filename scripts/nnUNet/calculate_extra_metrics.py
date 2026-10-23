"""
Compute Dice, IoU, sensitivity, specificity and HD95 for an nnU-Net
5-fold run, from each fold's validation/summary.json plus the predicted
and reference masks it points to.

    - Dice, IoU, TP, FP, FN, TN come straight from summary.json.
    - Sensitivity = TP / (TP + FN), specificity = TN / (TN + FP),
      derived from those counts.
    - HD95 is computed here from the masks (nnU-Net does not report it).

Aggregation: frame -> patient (mean over that patient's frames) ->
fold (mean over patients) -> mean +/- SD across folds. Frames where a
metric is undefined (e.g. empty reference AND empty prediction, or HD95
with an empty mask) are skipped for that metric via nanmean.

Usage:
    python nnunet_extra_metrics.py \
        --results_dir $nnUNet_results/Dataset001_X/nnUNetTrainer__nnUNetPlans__2d \
        --folds 0 1 2 3 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

LABEL = "1"   # foreground class key in summary.json


def load_mask(path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: float = 1.0) -> float:
    """95th percentile of the symmetric boundary-to-boundary distances.
    NaN if either mask is empty (distance undefined)."""
    if not pred.any() or not ref.any():
        return float("nan")
    pred_border = pred & ~ndi.binary_erosion(pred)
    ref_border = ref & ~ndi.binary_erosion(ref)
    dist_to_ref = ndi.distance_transform_edt(~ref_border)
    dist_to_pred = ndi.distance_transform_edt(~pred_border)
    d = np.concatenate([dist_to_ref[pred_border], dist_to_pred[ref_border]])
    return float(np.percentile(d, 95)) * spacing


def safe_div(a, b):
    return a / b if b > 0 else float("nan")


def case_rows(fold: int, results_dir: Path, spacing: float):
    summary_path = results_dir / f"fold_{fold}" / "validation" / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    rows = []
    for case in summary["metric_per_case"]:
        m = case["metrics"][LABEL]
        pred_file = Path(case["prediction_file"])
        ref_file = Path(case["reference_file"])
        case_id = pred_file.stem                      # e.g. LN003_frame00012

        row = {
            "fold": fold,
            "case": case_id,
            "patient": case_id.split("_")[0],
            "dice": m["Dice"],
            "iou": m["IoU"],
            "sensitivity": safe_div(m["TP"], m["TP"] + m["FN"]),
            "specificity": safe_div(m["TN"], m["TN"] + m["FP"]),
            "hd95": float("nan"),
        }
        if pred_file.exists() and ref_file.exists():
            row["hd95"] = hd95(load_mask(pred_file), load_mask(ref_file), spacing)
        else:
            print(f"  [warning] missing mask file(s) for {case_id}; HD95 skipped")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path,
                    help="folder containing fold_0 ... fold_4")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--spacing", type=float, default=1.0,
                    help="mm per pixel, to report HD95 in mm (default: pixels)")
    ap.add_argument("--out", default="nnunet_per_case_metrics.csv")
    args = ap.parse_args()

    rows = []
    for fold in args.folds:
        print(f"Fold {fold} ...")
        rows += case_rows(fold, args.results_dir, args.spacing)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    metrics = ["dice", "iou", "sensitivity", "specificity", "hd95"]

    # frame -> patient -> fold
    per_patient = df.groupby(["fold", "patient"])[metrics].mean()      # NaNs skipped
    per_fold = per_patient.groupby("fold").mean()
    print("\nPer-fold means (patient-averaged):")
    print(per_fold.round(4).to_string())

    mean, sd = per_fold.mean(), per_fold.std(ddof=1)
    print("\nMean +/- SD across folds:")
    for k in metrics:
        print(f"  {k:<12}{mean[k]:.4f} +/- {sd[k]:.4f}")

    fmt = lambda k, p: f"{mean[k]:.{p}f} $\\pm$ {sd[k]:.{p}f}"
    print("\nLaTeX row:")
    print("nnU-Net & " + " & ".join(
        [fmt("dice", 3), fmt("iou", 3), fmt("sensitivity", 3),
         fmt("specificity", 4), fmt("hd95", 1)]) + r" \\")

    n_empty = int(df["hd95"].isna().sum())
    print(f"\nFrames with undefined HD95 (empty pred or ref): {n_empty} / {len(df)}")
    print(f"Per-case results saved to {args.out}")


if __name__ == "__main__":
    main()