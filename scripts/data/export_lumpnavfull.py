"""
Export image/segmentation pairs from LumpNavFull (.npz) files to 2D PNGs.

For each patient LNXXX.npz, only the ultrasound frames that have a
corresponding painted segmentation are exported (identified via the
'index' array, which gives the position within 'image' for each frame
in 'segmentation').

Output structure (images and masks in parallel folders with matching
filenames -- the convention most segmentation frameworks, e.g. nnU-Net
style pipelines, expect):

    <EXPORT_FOLDER>/
        images/
            LN003_frame0012.png
            LN003_frame0047.png
            ...
        masks/
            LN003_frame0012.png
            LN003_frame0047.png
            ...

The "0012" in the filename is the frame's index within the *original*
'image' array (i.e. the value taken from 'index'), so you can always
trace an exported PNG back to its source frame.

Images and masks are converted to 8-bit (0-255) PNGs:
    - If already uint8, used as-is.
    - Otherwise min-max normalized per-frame to 0-255 (image).
    - Masks: if values are already in {0,1}, scaled to {0,255}; otherwise
      min-max normalized like the image. This assumes masks are binary
      or low-cardinality -- multi-class masks would need different
      handling (let me know if that's the case).

Memory handling: each patient's arrays are loaded once, only the
required frames are extracted and written immediately, and everything
is released + garbage-collected before moving to the next patient.
"""

import gc
import re
from pathlib import Path

import numpy as np
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavFull")
EXPORT_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNG")
IMAGES_SUBDIR = "images"
MASKS_SUBDIR = "masks"
# -----------------------------------------------------------------------

FNAME_RE = re.compile(r"^(?P<pid>LN\d+)\.npz$", re.IGNORECASE)
REQUIRED_KEYS = {"image", "segmentation", "index"}


def find_patient_files(folder: Path):
    files = {}
    for f in sorted(folder.glob("*.npz")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  [skip] Filename doesn't match expected pattern: {f.name}")
            continue
        files[m.group("pid")] = f
    return files


def to_uint8_image(frame: np.ndarray) -> np.ndarray:
    """Convert a single 2D image frame to uint8 via min-max normalization
    (no-op if already uint8)."""
    if frame.dtype == np.uint8:
        return frame
    frame = frame.astype(np.float32)
    mn, mx = float(frame.min()), float(frame.max())
    if mx - mn < 1e-9:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def to_uint8_mask(frame: np.ndarray) -> np.ndarray:
    """Convert a single 2D mask frame to a uint8 0/255 PNG-friendly array.
    Assumes binary or near-binary masks; scales {0,1} -> {0,255}."""
    frame = np.asarray(frame)
    mx = float(frame.max()) if frame.size else 0.0
    if mx <= 1.0:
        return (frame.astype(np.float32) * 255.0).astype(np.uint8)
    if frame.dtype == np.uint8:
        return frame
    # fallback: generic min-max normalization for non-binary masks
    mn = float(frame.min())
    if mx - mn < 1e-9:
        return np.zeros_like(frame, dtype=np.uint8)
    scaled = (frame.astype(np.float32) - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def export_patient(pid: str, path: Path, images_dir: Path, masks_dir: Path) -> dict:
    stats = {"patient_id": pid, "exported": 0, "skipped": 0, "error": ""}

    try:
        with np.load(path, allow_pickle=False) as data:
            missing_keys = REQUIRED_KEYS - set(data.keys())
            if missing_keys:
                stats["error"] = f"Missing expected key(s): {sorted(missing_keys)}"
                return stats

            img_stack = data["image"]
            seg_stack = data["segmentation"]
            index = data["index"]

            n_img_frames = img_stack.shape[0] if img_stack.ndim >= 3 else 1
            n_seg_frames = seg_stack.shape[0] if seg_stack.ndim >= 3 else 1

            if len(index) != n_seg_frames:
                print(
                    f"  [warning] {pid}: len(index)={len(index)} != "
                    f"segmentation frames={n_seg_frames}. Proceeding by "
                    f"pairing position-wise up to the shorter length."
                )

            n_pairs = min(len(index), n_seg_frames)

            for i in range(n_pairs):
                frame_idx = int(index[i])
                if frame_idx < 0 or frame_idx >= n_img_frames:
                    print(
                        f"  [skip] {pid}: index[{i}]={frame_idx} is out of "
                        f"range for image stack of length {n_img_frames}"
                    )
                    stats["skipped"] += 1
                    continue

                img_frame = to_uint8_image(img_stack[frame_idx])
                mask_frame = to_uint8_mask(seg_stack[i])

                fname = f"{pid}_frame{frame_idx:05d}.png"
                Image.fromarray(img_frame).save(images_dir / fname)
                Image.fromarray(mask_frame).save(masks_dir / fname)

                stats["exported"] += 1

                # Drop per-frame references promptly
                del img_frame, mask_frame

            # Drop the big stacks before the `with` block closes
            del img_stack, seg_stack, index
    except Exception as e:
        stats["error"] = f"Failed to load/process file: {e}"

    return stats


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    images_dir = EXPORT_FOLDER / IMAGES_SUBDIR
    masks_dir = EXPORT_FOLDER / MASKS_SUBDIR
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    patient_files = find_patient_files(SOURCE_FOLDER)
    print(f"Found {len(patient_files)} patient file(s) in {SOURCE_FOLDER}\n")

    total_exported = 0
    total_skipped = 0
    errors = []

    for pid, path in patient_files.items():
        print(f"Exporting {path.name} ...")
        stats = export_patient(pid, path, images_dir, masks_dir)
        if stats["error"]:
            print(f"  [error] {stats['error']}")
            errors.append((pid, stats["error"]))
        else:
            print(f"  exported {stats['exported']} pair(s), skipped {stats['skipped']}")
        total_exported += stats["exported"]
        total_skipped += stats["skipped"]

        # Free memory before moving to the next patient
        gc.collect()

    print("\n=== EXPORT SUMMARY ===")
    print(f"Total pairs exported: {total_exported}")
    print(f"Total frames skipped (bad index): {total_skipped}")
    if errors:
        print(f"\n{len(errors)} patient(s) failed:")
        for pid, err in errors:
            print(f"  {pid}: {err}")
    print(f"\nImages written to: {images_dir.resolve()}")
    print(f"Masks written to:  {masks_dir.resolve()}")


if __name__ == "__main__":
    main()