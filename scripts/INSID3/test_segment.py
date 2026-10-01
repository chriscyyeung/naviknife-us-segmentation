"""
Runs INSID3 on several random LumpNav images against one or more randomly
chosen reference pairs, and visualizes everything (reference(s) + each
random example's target image, ground-truth mask if found, and predicted
mask) in one grid image.
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
IMAGE_DIR = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images"
MASK_DIR = "/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks"
NUM_REFERENCES = 5                                  # how many random reference pairs to use
NUM_SAMPLES = 5                                     # how many random targets to segment
RANDOM_SEED = 42                                    # set to None for a different random sample each run
OUTPUT_PATH = "lumpnav_random_examples_grid.png"


def find_mask_for_image(image_path, mask_dir):
    """Looks for a mask file with the same basename as image_path inside
    mask_dir, trying a couple of common naming patterns. Returns None if
    nothing matches."""
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

    # Only images with a findable mask are eligible as references -- unlike
    # targets (where a missing ground-truth mask just leaves that grid cell
    # blank), a reference genuinely needs a real mask to be usable at all.
    images_with_masks = [p for p in all_images if find_mask_for_image(p, MASK_DIR) is not None]
    if len(images_with_masks) < NUM_REFERENCES:
        raise ValueError(
            f"Only found {len(images_with_masks)} images with a matching mask in {MASK_DIR}, "
            f"need at least {NUM_REFERENCES} to sample references from"
        )
    ref_image_paths = rng.sample(images_with_masks, NUM_REFERENCES)
    ref_mask_paths = [find_mask_for_image(p, MASK_DIR) for p in ref_image_paths]
    print(f"Selected {NUM_REFERENCES} random reference(s): "
          f"{[os.path.basename(p) for p in ref_image_paths]}")

    ref_basenames = {os.path.basename(p) for p in ref_image_paths}
    candidate_images = [p for p in all_images if os.path.basename(p) not in ref_basenames]
    if len(candidate_images) < NUM_SAMPLES:
        raise ValueError(
            f"Only found {len(candidate_images)} non-reference images in {IMAGE_DIR}, "
            f"need at least {NUM_SAMPLES}"
        )
    sample_image_paths = rng.sample(candidate_images, NUM_SAMPLES)

    # Build model (segment() internally clears its reference/target state via
    # reset_state() after every call, so the full reference set must be
    # re-added via set_reference() on every iteration below, not just once
    # before the loop)
    model = build_insid3()

    results = []  # (target_image_path, target_mask_path_or_None, pred_mask)
    for target_image_path in sample_image_paths:
        for ref_image_path, ref_mask_path in zip(ref_image_paths, ref_mask_paths):
            model.set_reference(ref_image_path, ref_mask_path)
        model.set_target(target_image_path)
        pred_mask = model.segment()
        target_mask_path = find_mask_for_image(target_image_path, MASK_DIR)
        results.append((target_image_path, target_mask_path, pred_mask))
        print(f"Segmented {target_image_path} "
              f"(ground truth {'found' if target_mask_path else 'NOT found'})")

    # ---- Build the grid: one row per reference pair, then one row per sampled example ----
    num_ref_rows = len(ref_image_paths)
    num_rows = num_ref_rows + len(results)
    num_cols = 3  # image, ground-truth/reference mask, prediction
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    for ref_row, (ref_image_path, ref_mask_path) in enumerate(zip(ref_image_paths, ref_mask_paths)):
        show(axes[ref_row, 0], ref_image_path, f"Reference {ref_row + 1}: {os.path.basename(ref_image_path)}")
        show(axes[ref_row, 1], ref_mask_path, f"Reference {ref_row + 1} mask", cmap="gray")
        axes[ref_row, 2].axis("off")  # nothing to put in the third column for reference rows

    for i, (target_image_path, target_mask_path, pred_mask) in enumerate(results):
        row = num_ref_rows + i
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