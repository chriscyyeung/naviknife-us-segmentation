#!/usr/bin/env python
"""Qualitative comparison figure: ultrasound frame, ground truth, and each model's mask.

One row per example, one column per panel:
    Image | Ground truth | nnU-Net | OpenUS | UltraSAM | INSID3 1-shot | INSID3 5-shot
          | INSID3+LoRA 1-shot | INSID3+LoRA 5-shot

Examples are drawn round-robin from the CV folds (example i comes from fold i % 5),
and each example is scored only by that fold's model, on that fold's own
validation patients. Only frames with a non-empty GT mask that every model has a
prediction for are eligible, because UltraSAM needs a GT box prompt and so cannot
predict empty frames. Within a fold, examples come from different patients where
possible.

Predictions at a different size (OpenUS, 224x224) are upsampled to GT size the
same way compute_seg_metrics.py does it: bilinear on 0/255, then threshold >127.

Example:
    python scripts/visualize_model_comparison.py --n_examples 5 --seed 0 \
        --out output/figures/model_comparison.png
"""
import argparse
import os
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/project/6106383/cyeung")
REPO = ROOT / "naviknife-us-segmentation"

# Literal {fold} placeholder, filled per fold.
DEFAULT_MODELS = [
    ("nnU-Net", str(ROOT / "nnUNet_results/Dataset067_LumpNav/nnUNetTrainer__nnUNetPlans__2d/fold_{fold}/validation")),
    ("OpenUS", str(REPO / "output/OpenUS/fold_{fold}/predicted_masks_teacher")),
    ("UltraSAM", str(REPO / "external/UltraSam/work_dirs/UltraSAM_box_refine_fold{fold}/pred_masks")),
    ("INSID3 1-shot", str(REPO / "external/INSID3/output/insid3_lumpnav/fold_{fold}/pred_masks")),
    ("INSID3 5-shot", str(REPO / "external/INSID3/output/insid3_lumpnav/fold_{fold}/pred_masks_5shot")),
    # Same INSID3, DINOv3 backbone LoRA-adapted per fold (train_lora_lumpnav.py)
    ("INSID3+LoRA 1-shot", str(REPO / "external/INSID3/output/insid3_lora/fold_{fold}/pred_masks_1shot")),
    ("INSID3+LoRA 5-shot", str(REPO / "external/INSID3/output/insid3_lora/fold_{fold}/pred_masks_5shot")),
]
DEFAULT_IMG_DIR = str(ROOT / "LumpNavPNGResMatchedCV/fold_{fold}/val/images")
DEFAULT_GT_DIR = str(ROOT / "LumpNavPNGResMatchedCV/fold_{fold}/val/masks")

GT_COLOR = (0.0, 0.85, 0.35)    # green
PRED_COLOR = (1.0, 0.25, 0.25)  # red


def load_mask(path, shape=None):
    """Binary mask from a PNG (bool, {0,1}, or 0/255), resized to `shape` (H, W) if needed."""
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8) * 255
    if shape is not None and arr.shape != tuple(shape):
        arr = np.array(Image.fromarray(arr).resize((shape[1], shape[0]), Image.BILINEAR))
        return arr > 127
    return arr > 0


def dice(pred, gt):
    denom = pred.sum() + gt.sum()
    return np.nan if denom == 0 else 2.0 * np.logical_and(pred, gt).sum() / denom


def overlay(ax, image, mask, color, alpha=0.4, outline=None, outline_color=GT_COLOR):
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if mask is not None and mask.any():
        rgba = np.zeros(mask.shape + (4,), dtype=float)
        rgba[mask] = (*color, alpha)
        ax.imshow(rgba, interpolation="nearest")
        ax.contour(mask, levels=[0.5], colors=[color], linewidths=1.0)
    if outline is not None and outline.any():
        ax.contour(outline, levels=[0.5], colors=[outline_color], linewidths=1.0, linestyles="--")
    ax.set_xticks([])
    ax.set_yticks([])


def eligible_frames(fold, args, models):
    gt_dir = Path(args.gt_dir.format(fold=fold))
    names = set(p.name for p in gt_dir.glob("*.png"))
    for _, pattern in models:
        names &= set(p.name for p in Path(pattern.format(fold=fold)).glob("*.png"))
    names = sorted(n for n in names if load_mask(gt_dir / n).sum() >= args.min_gt_area)
    return names


def pick_examples(args, models):
    rng = random.Random(args.seed)
    per_fold_count = {f: 0 for f in args.folds}
    order = [args.folds[i % len(args.folds)] for i in range(args.n_examples)]
    for f in order:
        per_fold_count[f] += 1

    picks = []
    pat = re.compile(args.patient_regex)
    for fold in args.folds:
        k = per_fold_count[fold]
        if k == 0:
            continue
        names = eligible_frames(fold, args, models)
        if not names:
            raise RuntimeError(f"fold {fold}: no frame has a non-empty GT and a prediction from every model")
        by_patient = {}
        for n in names:
            by_patient.setdefault(pat.match(n).group(1), []).append(n)
        patients = sorted(by_patient)
        rng.shuffle(patients)
        chosen = []
        # One frame per patient first, then reuse patients if k exceeds the patient count.
        while len(chosen) < k:
            for p in patients:
                remaining = [n for n in by_patient[p] if n not in chosen]
                if remaining and len(chosen) < k:
                    chosen.append(rng.choice(remaining))
            if all(not [n for n in by_patient[p] if n not in chosen] for p in patients):
                break
        print(f"fold {fold}: {len(names)} eligible frames from {len(patients)} patients -> {chosen}")
        picks.extend((fold, n) for n in chosen)
    # Interleave so rows run fold 0, 1, 2, 3, 4, 0, ...
    by_fold = {f: [n for ff, n in picks if ff == f] for f in args.folds}
    return [(f, by_fold[f].pop(0)) for f in order if by_fold[f]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_examples", type=int, default=5, help="Number of rows (examples cycle through the folds).")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Explicit examples as FOLD:NAME (e.g. 0:LN007_frame00029.png); overrides random selection.")
    ap.add_argument("--img_dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--gt_dir", default=DEFAULT_GT_DIR)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Override models as NAME=PRED_DIR (PRED_DIR uses a literal {fold}).")
    ap.add_argument("--patient_regex", default=r"(LN\d+)_frame")
    ap.add_argument("--min_gt_area", type=int, default=200, help="Minimum GT foreground pixels for a frame to be eligible.")
    ap.add_argument("--no_dice", action="store_true", help="Don't print per-panel Dice.")
    ap.add_argument("--panel_size", type=float, default=2.2, help="Panel width in inches.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default=str(REPO / "output/figures/model_comparison.png"),
                    help="Output path; a .pdf is written next to it too.")
    args = ap.parse_args()

    models = DEFAULT_MODELS
    if args.models:
        models = [tuple(m.split("=", 1)) for m in args.models]

    if args.cases:
        examples = [(int(c.split(":", 1)[0]), c.split(":", 1)[1]) for c in args.cases]
    else:
        examples = pick_examples(args, models)

    n_rows, n_cols = len(examples), 2 + len(models)
    # Frames are 615 (H) x 525 (W).
    aspect = 615 / 525
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(args.panel_size * n_cols, args.panel_size * aspect * n_rows + 0.6))

    titles = ["Image", "Ground truth"] + [m for m, _ in models]
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=10)

    for r, (fold, name) in enumerate(examples):
        image = np.array(Image.open(Path(args.img_dir.format(fold=fold)) / name).convert("L"))
        gt = load_mask(Path(args.gt_dir.format(fold=fold)) / name, image.shape)

        overlay(axes[r, 0], image, None, GT_COLOR)
        axes[r, 0].set_ylabel(f"Fold {fold}\n{Path(name).stem}", fontsize=8)
        overlay(axes[r, 1], image, gt, GT_COLOR)

        for c, (model, pattern) in enumerate(models, start=2):
            path = Path(pattern.format(fold=fold)) / name
            ax = axes[r, c]
            if not path.exists():
                overlay(ax, image, None, PRED_COLOR)
                ax.text(0.5, 0.5, "no prediction", color="w", ha="center", va="center", transform=ax.transAxes)
                continue
            pred = load_mask(path, image.shape)
            overlay(ax, image, pred, PRED_COLOR, outline=gt)
            if not args.no_dice:
                ax.text(0.03, 0.97, f"Dice {dice(pred, gt):.2f}", transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, color="w", bbox=dict(facecolor="black", alpha=0.6, pad=1.5, edgecolor="none"))

    fig.legend(handles=[Patch(facecolor=GT_COLOR, alpha=0.6, label="Ground truth (fill / dashed outline)"),
                        Patch(facecolor=PRED_COLOR, alpha=0.6, label="Model prediction")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.3, w_pad=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
