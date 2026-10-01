"""
Runs INSID3 on several random LumpNav images against a single fixed reference
pair, and visualizes everything (reference + each random example's target
image, ground-truth mask if found, and predicted mask) in one grid image.
"""

import os
import sys
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# INSID3's models/utils packages are local to its repo, not installed
# packages, so with this script living outside that repo, the repo's root
# needs to be added to sys.path before importing from them. Set via an env
# var (export INSID3_REPO_DIR=/path/to/INSID3 before running) so this
# doesn't need to be hardcoded/edited per machine.
INSID3_REPO_DIR = os.environ.get("INSID3_REPO_DIR", "/path/to/INSID3")  # TODO: set this
if not os.path.isdir(INSID3_REPO_DIR):
    raise FileNotFoundError(
        f"INSID3_REPO_DIR ({INSID3_REPO_DIR}) does not exist. Set the "
        f"INSID3_REPO_DIR environment variable to wherever you cloned the "
        f"INSID3 repo, e.g.: export INSID3_REPO_DIR=/home/you/repos/INSID3"
    )
sys.path.insert(0, INSID3_REPO_DIR)

from models import build_insid3
from utils.visualization import visualize_prediction_segmentation as visualize  # noqa: F401 (unused here, kept for parity with the original example)

# ---- Configuration -- adjust to match your LumpNav dataset layout ----
REF_IMAGE_PATH = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images/LN003_frame00000.png"
REF_MASK_PATH = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks/LN003_frame00000.png"
IMAGE_DIR = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images"
MASK_DIR = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks"
NUM_SAMPLES = 5
RANDOM_SEED = 42
OUTPUT_PATH = "lumpnav_random_examples_grid.png"


def find_mask_for_image(image_path, mask_dir):
    """Looks for a mask file with the same basename as image_path inside
    mask_dir, trying a couple of common naming patterns. Returns None if
    nothing matches (the grid just skips the ground-truth column for that
    row rather than failing)."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    candidates = (
        glob.glob(os.path.join(mask_dir, f"{basename}.*"))
        + glob.glob(os.path.join(mask_dir, f"{basename}_mask.*"))
    )
    return candidates[0] if candidates else None
 
 
def show(ax, image_or_path, title, cmap=None):
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path)
    elif hasattr(image_or_path, "detach"):  # torch.Tensor (possibly on GPU)
        image = image_or_path.detach().cpu().numpy()
    else:
        image = np.asarray(image_or_path)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
 
 
def main():
    rng = random.Random(RANDOM_SEED)
 
    all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    ref_basename = os.path.basename(REF_IMAGE_PATH)
    candidate_images = [p for p in all_images if os.path.basename(p) != ref_basename]
    if len(candidate_images) < len(all_images):
        print(f"Excluded reference image ({ref_basename}) from the sampling pool.")
    if len(candidate_images) < NUM_SAMPLES:
        raise ValueError(
            f"Only found {len(candidate_images)} non-reference images in {IMAGE_DIR}, "
            f"need at least {NUM_SAMPLES}"
        )
    sample_image_paths = rng.sample(candidate_images, NUM_SAMPLES)
 
    # Build model (segment() internally clears its reference/target state via
    # reset_state() after every call, so set_reference() must be called again
    # on every iteration below, not just once before the loop)
    model = build_insid3()
 
    results = []  # (target_image_path, target_mask_path_or_None, pred_mask)
    for target_image_path in sample_image_paths:
        model.set_reference(REF_IMAGE_PATH, REF_MASK_PATH)
        model.set_target(target_image_path)
        pred_mask = model.segment()
        target_mask_path = find_mask_for_image(target_image_path, MASK_DIR)
        results.append((target_image_path, target_mask_path, pred_mask))
        print(f"Segmented {target_image_path} "
              f"(ground truth {'found' if target_mask_path else 'NOT found'})")
 
    # ---- Build the grid: reference row, then one row per sampled example ----
    num_rows = 1 + len(results)
    num_cols = 3  # image, ground-truth mask (if available), prediction
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
 
    show(axes[0, 0], REF_IMAGE_PATH, "Reference image")
    show(axes[0, 1], REF_MASK_PATH, "Reference mask", cmap="gray")
    axes[0, 2].axis("off")  # nothing to put in the third column for the reference row
 
    for row, (target_image_path, target_mask_path, pred_mask) in enumerate(results, start=1):
        show(axes[row, 0], target_image_path, os.path.basename(target_image_path))
        if target_mask_path is not None:
            show(axes[row, 1], target_mask_path, "Ground truth", cmap="gray")
        else:
            axes[row, 1].axis("off")
        show(axes[row, 2], pred_mask, "Prediction", cmap="gray")
 
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved grid to {OUTPUT_PATH}")
    plt.show()
 
 
if __name__ == "__main__":
    main()
