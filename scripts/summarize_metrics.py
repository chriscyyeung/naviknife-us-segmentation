#!/usr/bin/env python
"""One comparison table from the per-frame CSVs written by compute_seg_metrics.py
(and nnUNet/calculate_extra_metrics.py).

Aggregation matches compute_seg_metrics.py: frame -> patient mean -> fold mean ->
mean +/- SD (ddof=1) across the 5 folds. NaN metrics (undefined, e.g. empty/empty)
are skipped in the means.

Two frame sets:
  all       every validation frame (UltraSAM has no predictions for empty-GT frames,
            so it is left out of this table)
  nonempty  frames whose GT mask contains tumour (the set every model can be scored on)

Usage (from scripts/):
    python summarize_metrics.py [--out_csv model_comparison_metrics.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = ["dice", "iou", "sensitivity", "specificity", "hd95"]

# (row label, per-frame CSV)
DEFAULT_MODELS = [
    ("nnU-Net", HERE / "nnUNet/nnunet_per_case_metrics.csv"),
    ("OpenUS", HERE / "openus_per_case_metrics.csv"),
    ("UltraSAM", HERE / "ultrasam_per_case_metrics.csv"),
    ("INSID3 1-shot", HERE / "insid3_1shot_metrics.csv"),
    ("INSID3 5-shot", HERE / "insid3_5shot_metrics.csv"),
    ("INSID3+LoRA 1-shot", HERE / "insid3_lora_1shot_metrics.csv"),
    ("INSID3+LoRA 5-shot", HERE / "insid3_lora_5shot_metrics.csv"),
]


def load(path, ref_empty_lookup):
    df = pd.read_csv(path)
    if "ref_empty" not in df:  # nnU-Net CSV: take GT emptiness from another model's CSV (same frames)
        df["ref_empty"] = df["case"].map(ref_empty_lookup)
        if df["ref_empty"].isna().any():
            raise ValueError(f"{path}: {df['ref_empty'].isna().sum()} frames with unknown GT emptiness")
        df["ref_empty"] = df["ref_empty"].astype(bool)
    return df


def aggregate(df):
    per_fold = df.groupby(["fold", "patient"])[METRICS].mean().groupby("fold").mean()
    return per_fold.mean(), per_fold.std(ddof=1), len(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_csv", default=str(HERE / "model_comparison_metrics.csv"))
    args = ap.parse_args()

    lookup_src = pd.read_csv(HERE / "openus_per_case_metrics.csv")
    lookup = dict(zip(lookup_src["case"], lookup_src["ref_empty"]))

    rows = []
    for frame_set in ["all", "nonempty"]:
        print(f"\n=== Frame set: {frame_set} "
              f"({'every validation frame' if frame_set == 'all' else 'GT contains tumour'}) ===")
        print(f"{'Model':<20}{'frames':>7}  " + "  ".join(f"{m:>17}" for m in METRICS))
        latex = []
        for name, path in DEFAULT_MODELS:
            if not Path(path).exists():
                print(f"{name:<20}  (missing {path})")
                continue
            df = load(path, lookup)
            if frame_set == "all" and df["ref_empty"].sum() == 0:
                print(f"{name:<20}  (no empty-GT frames scored; nonempty set only)")
                continue
            if frame_set == "nonempty":
                df = df[~df["ref_empty"]]
            mean, sd, n = aggregate(df)
            print(f"{name:<20}{n:>7}  " + "  ".join(
                f"{mean[m]:>8.{1 if m == 'hd95' else 3}f} ± {sd[m]:<6.{1 if m == 'hd95' else 3}f}" for m in METRICS))
            rows.append({"frame_set": frame_set, "model": name, "frames": n,
                         **{f"{m}_mean": mean[m] for m in METRICS}, **{f"{m}_sd": sd[m] for m in METRICS}})
            fmt = lambda m, p: f"{mean[m]:.{p}f} $\\pm$ {sd[m]:.{p}f}"
            latex.append(f"{name} & " + " & ".join(
                [fmt("dice", 3), fmt("iou", 3), fmt("sensitivity", 3), fmt("specificity", 4), fmt("hd95", 1)]) + r" \\")
        print("\nLaTeX rows:")
        print("\n".join(latex))

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
