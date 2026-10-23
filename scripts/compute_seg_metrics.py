"""
Model-agnostic segmentation metrics from SAVED PREDICTION MASKS.

Works for any model (OpenUS, UltraSAM, nnU-Net, ...) as long as you have,
per fold, a folder of predicted masks and a folder of ground-truth masks
with matching filenames. Every model is therefore scored by exactly the
same code, which is what makes the numbers comparable.

Metrics (per frame, on the ground-truth mask's native resolution):
    Dice, IoU, sensitivity, specificity, HD95
Conventions (same as nnU-Net's summary.json, so numbers are comparable):
    Dice  = 2TP / (2TP+FP+FN)      NaN if the denominator is 0
    IoU   = TP / (TP+FP+FN)        NaN if the denominator is 0
    sens  = TP / (TP+FN)           NaN if the reference is empty
    spec  = TN / (TN+FP)           NaN if there are no background pixels
    HD95  = 95th percentile of symmetric boundary distances;
            NaN if the prediction OR the reference is empty
NaN frames are skipped for that metric only.

Predictions at a different resolution than the ground truth (e.g. OpenUS
predicts at 224x224) are upsampled to the ground-truth size (bilinear on
the 0/255 mask, then re-thresholded) before scoring. This assumes the model
input was a plain resize of the full frame; if it was cropped or padded the
mapping back is not valid.

Aggregation: frame -> patient (mean of that patient's frames) -> fold
(mean over patients) -> mean +/- SD over folds. The pooled frame-level
mean over all frames is printed too, for comparison with tools that
report frame-level means.

Example (OpenUS):
    python compute_seg_metrics.py --model_name OpenUS \
        --pred_dir "/.../output/OpenUS/fold_{fold}/predicted_masks_teacher" \
        --gt_dir   "/.../LumpNavPNGResMatchedCV/fold_{fold}/val/masks" \
        --folds 0 1 2 3 4
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

METRICS = ["dice", "iou", "sensitivity", "specificity", "hd95"]


def load_mask(path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0          # works for {0,1}, {0,255} and boolean-mode PNGs


def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    """Bilinear-upsample a boolean mask to (H, W), then re-threshold."""
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(img) > 127


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: float = 1.0) -> float:
    if not pred.any() or not ref.any():
        return float("nan")
    pred_border = pred & ~ndi.binary_erosion(pred)
    ref_border = ref & ~ndi.binary_erosion(ref)
    dist_to_ref = ndi.distance_transform_edt(~ref_border)
    dist_to_pred = ndi.distance_transform_edt(~pred_border)
    d = np.concatenate([dist_to_ref[pred_border], dist_to_pred[ref_border]])
    return float(np.percentile(d, 95)) * spacing


def frame_metrics(pred: np.ndarray, ref: np.ndarray, spacing: float) -> dict:
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    tn = int((~pred & ~ref).sum())
    nan = float("nan")
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else nan,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else nan,
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else nan,
        "hd95": hd95(pred, ref, spacing),
        "ref_empty": not ref.any(),
        "pred_empty": not pred.any(),
    }


def find_pred(pred_dir: Path, gt_path: Path):
    p = pred_dir / gt_path.name
    if p.exists():
        return p
    matches = list(pred_dir.glob(gt_path.stem + ".*"))
    return matches[0] if matches else None


def fold_rows(fold, pred_dir, gt_dir, patient_re, spacing, only_nonempty_ref):
    rows, n_missing, n_resized = [], 0, 0
    gt_files = sorted(gt_dir.glob("*.png"))
    if not gt_files:
        print(f"  [warning] no PNGs in {gt_dir}")
    for gt_path in gt_files:
        pred_path = find_pred(pred_dir, gt_path)
        if pred_path is None:
            n_missing += 1
            continue
        ref = load_mask(gt_path)
        if only_nonempty_ref and not ref.any():
            continue
        pred = load_mask(pred_path)
        if pred.shape != ref.shape:
            pred = resize_mask(pred, ref.shape)
            n_resized += 1
        m = frame_metrics(pred, ref, spacing)
        m.update(fold=fold, case=gt_path.stem)
        pm = re.match(patient_re, gt_path.stem)
        m["patient"] = pm.group(1) if pm else gt_path.stem.split("_")[0]
        rows.append(m)
    print(f"  fold {fold}: {len(rows)} frames scored, {n_resized} resized to GT "
          f"size, {n_missing} GT frames without a prediction")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="Model")
    ap.add_argument("--pred_dir", required=True,
                    help="predicted-mask folder; use {fold} as a placeholder")
    ap.add_argument("--gt_dir", required=True,
                    help="ground-truth mask folder; use {fold} as a placeholder")
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--spacing", type=float, default=1.0,
                    help="mm per pixel for HD95 (default: report pixels)")
    ap.add_argument("--patient_regex", default=r"^(LN\d+)")
    ap.add_argument("--only_nonempty_ref", action="store_true",
                    help="score only frames whose ground truth contains tumor "
                         "(use this to match UltraSAM's frame set)")
    ap.add_argument("--out", default=None, help="per-case CSV path")
    args = ap.parse_args()

    rows = []
    for fold in args.folds:
        rows += fold_rows(fold, Path(args.pred_dir.format(fold=fold)),
                          Path(args.gt_dir.format(fold=fold)), args.patient_regex,
                          args.spacing, args.only_nonempty_ref)
    if not rows:
        raise SystemExit("No frames scored -- check --pred_dir / --gt_dir.")

    df = pd.DataFrame(rows)
    out = args.out or f"{args.model_name.lower().replace(' ', '_')}_per_case_metrics.csv"
    df.to_csv(out, index=False)

    per_patient = df.groupby(["fold", "patient"])[METRICS].mean()
    per_fold = per_patient.groupby("fold").mean()
    print("\nPer-fold means (patient-averaged):")
    print(per_fold.round(4).to_string())

    mean, sd = per_fold.mean(), per_fold.std(ddof=1)
    print("\nMean +/- SD across folds (patient-averaged):")
    for k in METRICS:
        print(f"  {k:<12}{mean[k]:.4f} +/- {sd[k]:.4f}")

    print(f"\nPooled over all {len(df)} frames (frame-level mean):")
    for k in METRICS:
        print(f"  {k:<12}{df[k].mean():.4f}")

    fmt = lambda k, p: f"{mean[k]:.{p}f} $\\pm$ {sd[k]:.{p}f}"
    print("\nLaTeX row (patient-averaged):")
    print(f"{args.model_name} & " + " & ".join(
        [fmt("dice", 3), fmt("iou", 3), fmt("sensitivity", 3),
         fmt("specificity", 4), fmt("hd95", 1)]) + r" \\")

    print(f"\nFrames: {len(df)} | empty reference: {int(df['ref_empty'].sum())} | "
          f"empty prediction: {int(df['pred_empty'].sum())} | "
          f"undefined HD95: {int(df['hd95'].isna().sum())}")
    print(f"Per-case results saved to {out}")


if __name__ == "__main__":
    main()