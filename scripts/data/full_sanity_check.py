"""
Sanity-check script for the LumpNavFull folder.

Each file is named LNXXX.npz and contains (at least) these keys:
    - image           : full ultrasound image stack (every acquired frame)
    - segmentation     : tumor segmentation mask stack (only painted frames,
                         typically a subset of the image frames)
    - contour          : (ignored)
    - imageToRef       : (ignored)
    - imageToNeedle     : (ignored)
    - index            : indices/reference into 'image' identifying which
                         frames were painted -> its length is expected to
                         match the number of 'segmentation' frames, NOT the
                         number of 'image' frames

For each patient this script checks:
    - Image resolution for 'image' and 'segmentation', and whether they match
      (spatial dimensions should agree even though frame counts differ)
    - Pixel intensity range (min/max/mean) for 'image'
    - Pixel value range for 'segmentation' (helps spot non-binary masks)
    - Number of empty masks (frames with no nonzero segmentation pixels)
    - Length of 'index' and whether it matches the number of segmentation
      frames (the expected relationship)
    - Reports the number of image frames as context, but does NOT flag
      image/segmentation frame-count differences as an error, since
      segmentation is expected to be a subset of image frames

Outputs a per-patient summary DataFrame, prints flags for anything odd,
and saves the summary to CSV.
"""

import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---- CONFIG ----------------------------------------------------------
FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavFull")           # change if your folder lives elsewhere
OUTPUT_CSV = "lumpnavfull_sanity_check.csv"
# -----------------------------------------------------------------------

FNAME_RE = re.compile(r"^(?P<pid>LN\d+)\.npz$", re.IGNORECASE)

REQUIRED_KEYS = {"image", "segmentation", "index"}


def count_empty_masks(seg_array: np.ndarray) -> int:
    """Count frames with no nonzero pixels. Assumes axis 0 is the frame
    axis for 3D+ arrays."""
    if seg_array.ndim <= 2:
        return int(np.count_nonzero(seg_array) == 0)
    return int(sum(np.count_nonzero(frame) == 0 for frame in seg_array))


def find_patient_files(folder: Path):
    files = {}
    for f in sorted(folder.glob("*.npz")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  [skip] Filename doesn't match expected pattern: {f.name}")
            continue
        files[m.group("pid")] = f
    return files


def analyze_patient(pid: str, path: Path) -> dict:
    row = {"patient_id": pid, "file": path.name}

    # Use a context manager so the underlying zip file handle is always
    # closed, and keep all big-array references scoped to this `with`
    # block so they go out of scope (and become garbage-collectible) as
    # soon as we're done with this patient.
    try:
        with np.load(path, allow_pickle=False) as data:
            missing_keys = REQUIRED_KEYS - set(data.keys())
            if missing_keys:
                row["error"] = f"Missing expected key(s): {sorted(missing_keys)}"
                return row

            img = data["image"]
            seg = data["segmentation"]
            idx = data["index"]

            # --- Image stats ---
            row["img_shape"] = img.shape
            row["img_num_frames"] = img.shape[0] if img.ndim >= 3 else 1
            row["img_resolution"] = img.shape[1:3] if img.ndim >= 3 else img.shape
            row["img_dtype"] = str(img.dtype)
            row["img_min"] = float(img.min())
            row["img_max"] = float(img.max())
            row["img_mean"] = round(float(img.mean()), 3)

            # --- Segmentation stats ---
            row["seg_shape"] = seg.shape
            row["seg_num_frames"] = seg.shape[0] if seg.ndim >= 3 else 1
            row["seg_resolution"] = seg.shape[1:3] if seg.ndim >= 3 else seg.shape
            row["seg_dtype"] = str(seg.dtype)
            row["seg_min"] = float(seg.min())
            row["seg_max"] = float(seg.max())
            row["num_empty_masks"] = count_empty_masks(seg)

            # --- Index stats ---
            row["index_len"] = len(idx) if hasattr(idx, "__len__") else None

            # --- Cross-checks / flags ---
            flags = []
            # NOTE: image frame count is expected to be >= segmentation frame
            # count (segmentation only covers painted/selected frames), so
            # this is not flagged as an error. It's still worth a heads-up
            # if segmentation somehow has MORE frames than image, which
            # would be a real problem.
            if row["seg_num_frames"] > row["img_num_frames"]:
                flags.append(
                    f"SEGMENTATION_EXCEEDS_IMAGE_FRAMES (img={row['img_num_frames']}, "
                    f"seg={row['seg_num_frames']})"
                )
            if row["img_resolution"] != row["seg_resolution"]:
                flags.append(
                    f"RESOLUTION_MISMATCH (img={row['img_resolution']}, "
                    f"seg={row['seg_resolution']})"
                )
            if row["index_len"] is not None and row["index_len"] != row["seg_num_frames"]:
                flags.append(
                    f"INDEX_LENGTH_MISMATCH (index={row['index_len']}, "
                    f"seg={row['seg_num_frames']})"
                )

            row["flags"] = "; ".join(flags) if flags else ""
            row["error"] = ""

            # Explicitly drop references to the big arrays before leaving
            # the `with` block (and before the NpzFile itself is closed),
            # so nothing lingers on the heap longer than necessary.
            del img, seg, idx
    except Exception as e:
        row["error"] = f"Failed to load/process file: {e}"
        return row

    return row


def main():
    if not FOLDER.exists():
        print(f"Folder not found: {FOLDER.resolve()}")
        return

    patient_files = find_patient_files(FOLDER)
    print(f"Found {len(patient_files)} patient file(s) in {FOLDER}\n")

    rows = []
    for pid, path in patient_files.items():
        print(f"Processing {path.name} ...")
        rows.append(analyze_patient(pid, path))
        # Force garbage collection after each file. np.load() on a .npz
        # lazily decompresses each array into memory the first time it's
        # accessed, and large ultrasound stacks can be sizeable (hundreds
        # of MB to a few GB). Without an explicit collect(), Python's
        # reference-counting usually frees these immediately, but cyclic
        # references held internally by NpzFile / numpy can delay
        # collection until the next gc pass, which adds up across many
        # patients and can trigger an OOM kill.
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    errored = df[df["error"] != ""] if "error" in df.columns else pd.DataFrame()
    if not errored.empty:
        print(f"\n{len(errored)} patient(s) failed to load / parse:")
        print(errored[["patient_id", "error"]].to_string(index=False))

    if "flags" in df.columns:
        flagged = df[(df["flags"] != "") & (df["error"] == "")]
        print(f"\n{len(flagged)} patient(s) with issues flagged:")
        if not flagged.empty:
            print(flagged[["patient_id", "flags"]].to_string(index=False))
        else:
            print("  None — all patients passed the sanity checks.")

    if "num_empty_masks" in df.columns:
        total_empty = df["num_empty_masks"].dropna().sum()
        print(f"\nTotal empty masks across all patients: {int(total_empty)}")

    # Check resolution consistency across the whole dataset
    if "img_resolution" in df.columns:
        all_res = pd.concat([df["img_resolution"], df["seg_resolution"]]).dropna().unique()
        if len(all_res) > 1:
            print(f"\n[WARNING] Multiple distinct resolutions found across dataset: {list(all_res)}")
        elif len(all_res) == 1:
            print(f"\nAll images share a consistent resolution: {all_res[0]}")

    print(f"\nFull results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()