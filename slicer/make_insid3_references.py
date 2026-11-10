"""
Pick a fixed INSID3 reference set for insid3_adapter.py and copy it into
<model_dir>/references/{images,masks}/.

Samples --shots frames with a non-empty tumour mask, at most one per patient, so
the references cover different tumours. Same rule as the LumpNav evaluation
(non-empty frames only), except the set is fixed instead of re-drawn per frame.

The simulation cases (NaviKnife / LumpNav2) are not LumpNav patients, so any
LumpNav patient can be a reference. To evaluate on LumpNav frames instead, pass
--exclude-patients with the patients being segmented.

    python make_insid3_references.py \
        --image-dir D:/.../LumpNavPNGResMatched/images \
        --mask-dir  D:/.../LumpNavPNGResMatched/masks \
        --model-dir C:/.../models/INSID3_LoRA --shots 5
"""

import argparse
import glob
import os
import re
import shutil

import numpy as np
from PIL import Image

_FRAME_RE = re.compile(r"^(LN\d+)_frame(\d+)$")


def frame_name(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return name[:-5] if name.endswith("_0000") else name


def mask_area(path):
    mask = np.array(Image.open(path)) > 0
    return int(mask.any(axis=-1).sum() if mask.ndim == 3 else mask.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--model-dir", required=True, help="writes <model-dir>/references/{images,masks}")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-area-percentile", type=float, default=0.0,
                   help="only use frames whose tumour area is at least this percentile of non-empty "
                        "frames (0 = any non-empty frame, as in the evaluation)")
    p.add_argument("--exclude-patients", nargs="*", default=[], help="e.g. LN003 LN010")
    args = p.parse_args()

    images = {frame_name(f): f for f in glob.glob(os.path.join(args.image_dir, "*.png"))}
    masks = {frame_name(f): f for f in glob.glob(os.path.join(args.mask_dir, "*.png"))}
    excluded = set(args.exclude_patients)

    candidates = []  # (patient, name, area)
    for name in sorted(set(images) & set(masks)):
        match = _FRAME_RE.match(name)
        if not match or match.group(1) in excluded:
            continue
        area = mask_area(masks[name])
        if area > 0:
            candidates.append((match.group(1), name, area))
    if not candidates:
        raise SystemExit("No non-empty frames found; check --image-dir/--mask-dir")

    if args.min_area_percentile > 0:
        cutoff = np.percentile([a for _, _, a in candidates], args.min_area_percentile)
        candidates = [c for c in candidates if c[2] >= cutoff]

    by_patient = {}
    for patient, name, _ in candidates:
        by_patient.setdefault(patient, []).append(name)
    patients = sorted(by_patient)
    if len(patients) < args.shots:
        raise SystemExit(f"Only {len(patients)} patients have eligible frames; need {args.shots}")

    rng = np.random.default_rng(args.seed)
    chosen = [by_patient[pt][rng.integers(len(by_patient[pt]))]
              for pt in sorted(rng.choice(patients, args.shots, replace=False))]

    out = os.path.join(args.model_dir, "references")
    for sub in ("images", "masks"):
        if os.path.isdir(os.path.join(out, sub)):
            shutil.rmtree(os.path.join(out, sub))
        os.makedirs(os.path.join(out, sub))
    for name in chosen:
        shutil.copy(images[name], os.path.join(out, "images", name + ".png"))
        # Save as 0/255 single channel so the adapter reads it the same way however the source was stored
        mask = np.array(Image.open(masks[name])) > 0
        mask = mask.any(axis=-1) if mask.ndim == 3 else mask
        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(out, "masks", name + ".png"))
    print(f"Wrote {len(chosen)} references to {out}: {chosen}")


if __name__ == "__main__":
    main()
