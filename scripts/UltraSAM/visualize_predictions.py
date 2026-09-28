"""
Run UltraSAM inference on N frames and show them together in one grid plot.

Each panel shows:
    - the input frame
    - the highest-confidence predicted mask (red tint)
    - the prompts actually given to the model: box (green), point (yellow)
    - optionally the ground-truth outline (cyan) and Dice, if --mask_dir is given

The model is loaded once and reused for every frame. Uses the non-interactive
Agg backend so it works over SSH (saves to a file instead of opening a window).

Example:
    python inference_multi_frame_grid.py --num_frames 8 --patient LN003
    python inference_multi_frame_grid.py --num_frames 12 --cols 4 \
        --mask_dir /path/to/masks_boolean --out grid.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend (no display over SSH)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import torch.nn.functional as F
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from endosam.models.utils.custom_functional import multi_head_attention_forward

# Monkey patch the multi_head_attention_forward function
F.multi_head_attention_forward = multi_head_attention_forward

# ---- DEFAULTS (override on the command line) -------------------------------
DEFAULT_CONFIG = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_inference.py"
DEFAULT_CKPT = "/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/external/UltraSam/work_dirs/UltraSAM_box_refine_fold0/best_coco_segm_mAP_iter_25000.pth"
DEFAULT_IMAGE_DIR = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV/fold_0/val/images"
# -----------------------------------------------------------------------------


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def extract_prompts(result):
    """Prompts fed to the model, converted back to ORIGINAL image pixel
    coordinates (they are stored in resized-image space: original * scale + 0.5)."""
    gt = result.gt_instances
    sx, sy = result.metainfo["scale_factor"]

    def to_original(xy):
        out = xy.astype(np.float64).copy()
        out[..., 0] = (out[..., 0] - 0.5) / sx
        out[..., 1] = (out[..., 1] - 0.5) / sy
        return out

    boxes = points = None
    if "boxes" in gt:
        boxes = to_original(to_numpy(gt.boxes).reshape(-1, 2, 2))
    if "points" in gt:
        points = to_original(to_numpy(gt.points).reshape(-1, 2))
    return boxes, points


def dice_score(pred, gt):
    denom = pred.sum() + gt.sum()
    return 2.0 * np.logical_and(pred, gt).sum() / denom if denom > 0 else float("nan")


def load_gt_mask(mask_dir, frame_name):
    """Load a ground-truth mask by matching filename. Returns bool array or None."""
    if mask_dir is None:
        return None
    path = Path(mask_dir) / frame_name
    if not path.exists():
        return None
    arr = np.array(Image.open(path))        # boolean-mode PNGs load as bool
    if arr.ndim == 3:
        arr = np.array(Image.open(path).convert("L"))
    return arr.astype(bool) if arr.dtype == bool else arr > 0


def select_frames(image_dir, patient, n, mode):
    pattern = f"{patient}_*.png" if patient else "*.png"
    files = sorted(Path(image_dir).glob(pattern))
    if not files:
        raise SystemExit(f"No images matching '{pattern}' in {image_dir}")
    n = min(n, len(files))
    if mode == "first":
        idx = np.arange(n)
    else:  # evenly spaced across the whole sequence
        idx = np.linspace(0, len(files) - 1, n).round().astype(int)
    return [files[i] for i in idx]


def run_frame(model, path, mask_dir):
    arr = np.array(Image.open(path).convert("RGB"))
    img_h, img_w = arr.shape[:2]
    result = inference_detector(model, arr)

    boxes, points = extract_prompts(result)

    pred, score = None, None
    masks = result.pred_instances.masks
    if len(masks) > 0:
        masks = to_numpy(masks)
        scores = to_numpy(result.pred_instances.scores)
        best = int(scores.argmax())
        pred, score = masks[best].astype(bool), float(scores[best])

    gt = load_gt_mask(mask_dir, path.name)
    dice = None
    if pred is not None and gt is not None and pred.shape == gt.shape:
        dice = dice_score(pred, gt)

    # One-line console sanity check per frame
    msg = f"{path.name}: "
    msg += f"score={score:.3f}" if score is not None else "no prediction"
    if dice is not None:
        msg += f"  dice={dice:.3f}"
    if boxes is not None:
        (x1, y1), (x2, y2) = boxes[0]
        cov = max(x2 - x1, 0) * max(y2 - y1, 0) / (img_w * img_h)
        msg += f"  box_coverage={cov:.0%}"
        if cov < 0.5 or cov > 1.1:
            msg += "  [WARNING: full-size box looks wrong]"
    print(msg)

    return dict(name=path.name, image=arr, pred=pred, score=score,
                gt=gt, dice=dice, boxes=boxes, points=points)


def plot_grid(records, cols, out_path, title):
    n = len(records)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    h, w = records[0]["image"].shape[:2]
    panel_w = 4.0
    panel_h = panel_w * h / w

    fig, axes = plt.subplots(rows, cols, figsize=(cols * panel_w, rows * (panel_h + 0.4)))
    axes = np.atleast_1d(axes).ravel()

    for ax, rec in zip(axes, records):
        img_h, img_w = rec["image"].shape[:2]
        ax.imshow(rec["image"])

        if rec["pred"] is not None:
            overlay = np.zeros((*rec["pred"].shape, 4))
            overlay[rec["pred"]] = (1.0, 0.0, 0.0, 0.4)
            ax.imshow(overlay)

        if rec["gt"] is not None and rec["gt"].any():
            ax.contour(rec["gt"].astype(float), levels=[0.5], colors="cyan", linewidths=1.2)

        if rec["boxes"] is not None:
            for (x1, y1), (x2, y2) in rec["boxes"]:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor="lime", linewidth=2))
        if rec["points"] is not None:
            for x, y in rec["points"]:
                ax.plot(x, y, marker="o", markersize=7, markerfacecolor="yellow",
                        markeredgecolor="black")

        # Keep the view on the image even if a prompt box extends past its edges
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis("off")

        t = rec["name"]
        if rec["score"] is not None:
            t += f"\nscore {rec['score']:.2f}"
            if rec["dice"] is not None:
                t += f" | dice {rec['dice']:.2f}"
        else:
            t += "\nno prediction"
        ax.set_title(t, fontsize=9)

    for ax in axes[n:]:   # hide unused panels in the last row
        ax.axis("off")

    handles = [
        Line2D([0], [0], color="lime", lw=2, label="box prompt"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markeredgecolor="black", label="point prompt"),
        Line2D([0], [0], color="red", lw=6, alpha=0.4, label="prediction"),
        Line2D([0], [0], color="cyan", lw=2, label="ground truth"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, default=8, help="how many frames to plot")
    p.add_argument("--patient", default=None, help="only use frames from this patient, e.g. LN003")
    p.add_argument("--select", choices=["even", "first"], default="even",
                   help="'even' = evenly spaced across the sequence, 'first' = first N frames")
    p.add_argument("--cols", type=int, default=4, help="grid columns")
    p.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--mask_dir", default=None,
                   help="optional GT masks folder (same filenames) for outline + Dice")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="./work_dirs/inference_grid.png")
    args = p.parse_args()

    frames = select_frames(args.image_dir, args.patient, args.num_frames, args.select)
    print(f"Running inference on {len(frames)} frame(s)\n")

    model = init_detector(args.config, args.checkpoint, device=args.device)

    records = [run_frame(model, path, args.mask_dir) for path in frames]

    title = f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}"
    plot_grid(records, args.cols, args.out, title)
    print(f"\nSaved grid to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()