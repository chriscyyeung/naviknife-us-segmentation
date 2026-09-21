"""
Generate a COCO-format annotations JSON from a folder with "images" and
"masks" subfolders (matching filenames in both), for use with UltraSAM
or other COCO-compatible instance segmentation tools.

Assumptions:
    - LumpNavPNGResMatched/images/<name>.png : grayscale ultrasound image
    - LumpNavPNGResMatched/masks_boolean/<name>.png : boolean mask (PIL
      mode "1" -- True = foreground/tumor, False = background). Loaded
      WITHOUT .convert("L"), so it comes back as a native bool array
      directly -- no thresholding needed.
    - Single class: "tumor" (category_id = 1)
    - A mask may contain multiple disconnected tumor regions -- each
      connected component becomes its own COCO annotation (standard
      instance-segmentation behavior)
    - Images with an all-background (empty) mask are still included in
      the "images" list, just with zero annotations -- this is normal,
      valid COCO behavior for negative examples

Segmentation encoding:
    - Uses RLE (run-length encoding), the standard COCO representation
      for pixel masks (as opposed to polygon vertices), which is what
      SAM-derived models typically expect since it handles arbitrary
      (non-convex, disconnected, or holed) shapes cleanly.
    - Uses pycocotools for proper compressed RLE if installed
      (`pip install pycocotools`) -- strongly recommended, since this
      matches what most COCO-consuming tools expect.
    - Falls back to a manual *uncompressed* RLE encoding (still valid
      per the COCO spec) if pycocotools isn't available, so the script
      still works without extra dependencies, just less compactly.

Output: <OUTPUT_JSON> written directly in DATASET_FOLDER by default.
"""

import json
import re
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

try:
    from pycocotools import mask as mask_utils
    HAVE_PYCOCOTOOLS = True
except ImportError:
    HAVE_PYCOCOTOOLS = False

# ---- CONFIG ----------------------------------------------------------
DATASET_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched")
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
OUTPUT_JSON = DATASET_FOLDER / "coco_annotations.json"
CATEGORY_NAME = "tumor"
MIN_COMPONENT_AREA = 5     # ignore tiny connected components (likely noise)
# -----------------------------------------------------------------------


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a mask as a uint8 {0,1} array. Handles boolean-mode (PIL "1")
    masks natively -- no .convert("L") or thresholding needed, since
    boolean masks are already exactly foreground/background. Falls back
    to thresholding for any non-boolean (e.g. legacy 0/255) masks, so
    this stays compatible with older mask folders too."""
    with Image.open(mask_path) as im:
        arr = np.array(im)

    if arr.dtype == bool:
        return arr.astype(np.uint8)

    # Fallback for non-boolean masks (e.g. 0/255 grayscale)
    if arr.ndim == 3:
        arr = np.array(Image.open(mask_path).convert("L"))
    return (arr > 0).astype(np.uint8)
# -----------------------------------------------------------------------


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    """Encode a boolean/0-1 mask as COCO RLE. Uses pycocotools if
    available (compressed RLE), otherwise a manual uncompressed RLE."""
    if HAVE_PYCOCOTOOLS:
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        # 'counts' is bytes -- COCO JSON requires a string
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    else:
        # Manual uncompressed RLE (COCO spec allows this: counts is a
        # plain list of run lengths, column-major / Fortran order,
        # starting with a run of 0s).
        flat = binary_mask.T.flatten()  # column-major (Fortran) order
        counts = []
        prev_val = 0
        run_length = 0
        for val in flat:
            if val == prev_val:
                run_length += 1
            else:
                counts.append(run_length)
                prev_val = val
                run_length = 1
        counts.append(run_length)
        return {"size": [binary_mask.shape[0], binary_mask.shape[1]], "counts": counts}


def mask_area_and_bbox(binary_mask: np.ndarray):
    ys, xs = np.where(binary_mask)
    if len(ys) == 0:
        return 0, [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    area = int(binary_mask.sum())
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # COCO: [x,y,width,height]
    return area, bbox


def main():
    images_dir = DATASET_FOLDER / IMAGES_SUBDIR
    masks_dir = DATASET_FOLDER / MASKS_SUBDIR

    if not images_dir.exists():
        print(f"Images folder not found: {images_dir.resolve()}")
        return
    if not masks_dir.exists():
        print(f"Masks folder not found: {masks_dir.resolve()}")
        return

    if not HAVE_PYCOCOTOOLS:
        print(
            "[warning] pycocotools not installed -- falling back to manual "
            "uncompressed RLE. Run `pip install pycocotools` for standard, "
            "more compact compressed RLE (recommended for UltraSAM "
            "compatibility)."
        )

    image_files = sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} image(s) in {images_dir}\n")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": CATEGORY_NAME, "supercategory": "none"}],
    }

    image_id = 1
    annotation_id = 1
    n_empty_masks = 0
    n_multi_instance = 0
    n_missing_masks = 0

    for img_path in image_files:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            print(f"  [warning] No matching mask for {img_path.name}, skipping")
            n_missing_masks += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        binary_mask = load_binary_mask(mask_path)

        if binary_mask.max() == 0:
            n_empty_masks += 1
            image_id += 1
            continue

        # Separate disconnected tumor regions into individual instances
        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        instance_count_this_image = 0

        for label_id in range(1, num_labels):  # label 0 is background
            component_mask = (labels == label_id)
            area, bbox = mask_area_and_bbox(component_mask)

            if area < MIN_COMPONENT_AREA:
                continue

            rle = encode_mask_rle(component_mask)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            annotation_id += 1
            instance_count_this_image += 1

        if instance_count_this_image > 1:
            n_multi_instance += 1

        image_id += 1

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f)

    print("=== SUMMARY ===")
    print(f"Images written: {len(coco['images'])}")
    print(f"Annotations written: {len(coco['annotations'])}")
    print(f"Images with empty (no tumor) masks: {n_empty_masks}")
    print(f"Images with multiple tumor instances: {n_multi_instance}")
    if n_missing_masks:
        print(f"Images skipped (no matching mask found): {n_missing_masks}")
    print(f"\nCOCO annotations saved to: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()