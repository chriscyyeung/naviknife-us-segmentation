"""
Generate a COCO-format annotations JSON for every fold's train/ and val/
split, from a 5-fold CV directory structure like:

    <CV_ROOT>/
        fold_0/
            train/{images,masks_boolean}/
            val/{images,masks_boolean}/
        fold_1/
            ...

For each fold_i/{train,val} folder, writes a coco_annotations.json
directly inside that split folder (e.g. fold_0/train/coco_annotations.json),
so it sits alongside its own images/ and masks_boolean/ subfolders --
matching the data_root / data_prefix structure mmdet's CocoDataset
config expects.

Same behavior as the earlier "nonempty" generator: images with an empty
(all-background) mask are excluded entirely, since UltraSAM's
prompt-based training/testing has nothing to generate a prompt from for
those images. Boolean-mode masks are read directly (no thresholding
needed); falls back to thresholding for legacy 0/255 masks.

A val split's JSON can be reused directly as a fold's test_dataloader
ann_file too -- no separate generation needed, since val already serves
as that fold's held-out evaluation set in a k-fold CV setup.
"""

import json
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
CV_ROOT = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV")
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
OUTPUT_JSON_NAME = "coco_annotations.json"
SPLITS = ["train", "val"]
CATEGORY_NAME = "tumor"
MIN_COMPONENT_AREA = 5
# -----------------------------------------------------------------------


def load_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as im:
        arr = np.array(im)
    if arr.dtype == bool:
        return arr.astype(np.uint8)
    if arr.ndim == 3:
        arr = np.array(Image.open(mask_path).convert("L"))
    return (arr > 0).astype(np.uint8)


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    if HAVE_PYCOCOTOOLS:
        rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    else:
        flat = binary_mask.T.flatten()
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
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return area, bbox


def generate_coco_for_split(split_dir: Path) -> dict:
    """Generate and write a COCO JSON for one fold's train/ or val/
    folder. Returns a small stats dict for the summary printout."""
    images_dir = split_dir / IMAGES_SUBDIR
    masks_dir = split_dir / MASKS_SUBDIR
    output_path = split_dir / OUTPUT_JSON_NAME

    stats = {"split_dir": split_dir, "n_included": 0, "n_excluded_empty": 0,
              "n_missing_masks": 0, "n_annotations": 0, "n_multi_instance": 0}

    if not images_dir.exists():
        print(f"  [skip] {images_dir} does not exist")
        return stats
    if not masks_dir.exists():
        print(f"  [skip] {masks_dir} does not exist")
        return stats

    image_files = sorted(images_dir.glob("*.png"))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": CATEGORY_NAME, "supercategory": "none"}],
    }

    image_id = 1
    annotation_id = 1

    for img_path in image_files:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            stats["n_missing_masks"] += 1
            continue

        binary_mask = load_binary_mask(mask_path)

        if binary_mask.max() == 0:
            stats["n_excluded_empty"] += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        instance_count_this_image = 0

        for label_id in range(1, num_labels):
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
            stats["n_multi_instance"] += 1

        stats["n_included"] += 1
        image_id += 1

    stats["n_annotations"] = len(coco["annotations"])

    with open(output_path, "w") as f:
        json.dump(coco, f)

    return stats


def main():
    if not CV_ROOT.exists():
        print(f"CV root not found: {CV_ROOT.resolve()}")
        return

    if not HAVE_PYCOCOTOOLS:
        print(
            "[warning] pycocotools not installed -- falling back to manual "
            "uncompressed RLE. Run `pip install pycocotools` for standard "
            "compressed RLE (recommended for UltraSAM compatibility).\n"
        )

    fold_dirs = sorted(CV_ROOT.glob("fold_*"))
    print(f"Found {len(fold_dirs)} fold folder(s) in {CV_ROOT}\n")

    all_stats = []

    for fold_dir in fold_dirs:
        print(f"=== {fold_dir.name} ===")
        for split in SPLITS:
            split_dir = fold_dir / split
            print(f"  Processing {split_dir} ...")
            stats = generate_coco_for_split(split_dir)
            all_stats.append((fold_dir.name, split, stats))
            if stats["n_included"] or stats["n_excluded_empty"]:
                print(f"    included={stats['n_included']}  "
                      f"excluded_empty={stats['n_excluded_empty']}  "
                      f"annotations={stats['n_annotations']}  "
                      f"multi_instance={stats['n_multi_instance']}")

    print("\n=== SUMMARY ===")
    print(f"{'fold':<10}{'split':<8}{'included':<10}{'excl_empty':<12}"
          f"{'annotations':<13}{'multi_inst':<10}")
    for fold_name, split, stats in all_stats:
        print(f"{fold_name:<10}{split:<8}{stats['n_included']:<10}"
              f"{stats['n_excluded_empty']:<12}{stats['n_annotations']:<13}"
              f"{stats['n_multi_instance']:<10}")

    print(
        f"\nEach fold_i/train/{OUTPUT_JSON_NAME} and fold_i/val/{OUTPUT_JSON_NAME} "
        "written directly inside its respective split folder.\n"
        f"For test_dataloader, reuse each fold's val/{OUTPUT_JSON_NAME} directly -- "
        "no separate test-set generation needed for a CV setup."
    )


if __name__ == "__main__":
    main()