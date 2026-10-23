#!/usr/bin/env python
"""Convert UltraSAM (mmdet CocoMetric format_only) segm results to per-frame
binary mask PNGs, so they can be scored with compute_seg_metrics.py.

Inputs:
  --gt_coco   the fold's val coco_annotations.json (gives image_id -> file_name, size)
  --pred_json the *.segm.json written by CocoMetric(format_only=True, outfile_prefix=...)
Output:
  one 0/255 PNG per GT image in --out_dir, named after the image's file_name stem.
  All instances above --score_thr are unioned (one tumor class). Images with no
  prediction get an empty mask, so nothing is silently skipped at scoring time.

Example (per fold):
  python coco_preds_to_masks.py \
      --gt_coco  .../fold_0/val/coco_annotations.json \
      --pred_json .../fold_0/ultrasam_preds.segm.json \
      --out_dir  .../fold_0/predicted_masks_ultrasam
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode(seg, h, w):
    if isinstance(seg, list):  # polygon
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(seg["counts"], list):  # uncompressed RLE
        rle = mask_utils.frPyObjects(seg, h, w)
    else:  # compressed RLE (mmdet default); counts may be str
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_coco", required=True)
    p.add_argument("--pred_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--score_thr", type=float, default=0.0)
    p.add_argument("--suffix", default="", help="appended to the file stem, e.g. '_mask'")
    args = p.parse_args()

    with open(args.gt_coco) as f:
        images = {im["id"]: im for im in json.load(f)["images"]}
    with open(args.pred_json) as f:
        preds = json.load(f)

    by_image = defaultdict(list)
    unknown = 0
    for det in preds:
        if det["image_id"] not in images:
            unknown += 1
            continue
        if det.get("score", 1.0) >= args.score_thr:
            by_image[det["image_id"]].append(det)

    os.makedirs(args.out_dir, exist_ok=True)
    n_empty = 0
    size_mismatch = 0
    for img_id, im in images.items():
        h, w = im["height"], im["width"]
        out = np.zeros((h, w), dtype=bool)
        for det in by_image.get(img_id, []):
            m = decode(det["segmentation"], h, w)
            if m.shape != out.shape:
                size_mismatch += 1
                continue
            out |= m
        if not out.any():
            n_empty += 1
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        Image.fromarray(out.astype(np.uint8) * 255).save(
            os.path.join(args.out_dir, stem + args.suffix + ".png"))

    print(f"{len(images)} masks written to {args.out_dir}; "
          f"{n_empty} empty, {len(preds)} detections read, "
          f"{unknown} with unknown image_id, {size_mismatch} size mismatches skipped")


if __name__ == "__main__":
    main()
