"""
Sanity-check script for PatientArrays folder.

Expects files named like:
    LN001_ultrasound.npy
    LN001_segmentation.npy

For each patient it checks:
    - Whether both ultrasound and segmentation files exist
    - Whether the number of images (frames) matches between the two
    - Image resolution (H x W) for both, and whether they match
    - Pixel value range (min/max) for the ultrasound images
    - Pixel value range for the segmentation masks (useful for spotting
      non-binary masks)
    - Number of "empty" masks (frames where the segmentation is all zero /
      no tumor annotated)

Outputs a per-patient summary table (as a pandas DataFrame) and prints
flags for anything that looks wrong. Also saves the summary to a CSV.
"""

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- CONFIG ----------------------------------------------------------
FOLDER = Path("/home/cyeung/projects/aip-medilab/shared/LumpNavPatientArrays")   # change if your folder lives elsewhere
OUTPUT_CSV = "patient_array_sanity_check.csv"
# -----------------------------------------------------------------------

FNAME_RE = re.compile(r"^(?P<pid>.+?)_(?P<kind>ultrasound|segmentation)\.npy$")


def find_patient_files(folder: Path):
    """Group files in folder by patient ID -> {'ultrasound': path, 'segmentation': path}"""
    patients = defaultdict(dict)
    for f in sorted(folder.glob("*.npy")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  [skip] Filename doesn't match expected pattern: {f.name}")
            continue
        pid = m.group("pid")
        kind = m.group("kind")
        patients[pid][kind] = f
    return patients


def load_array(path: Path):
    try:
        return np.load(path)
    except Exception as e:
        print(f"  [error] Failed to load {path.name}: {e}")
        return None


def count_empty_masks(seg_array: np.ndarray) -> int:
    """
    Count frames where the mask has no positive (nonzero) pixels.
    Assumes the first axis is the image/frame axis for a 3D array.
    """
    if seg_array.ndim == 2:
        # single image, not a stack
        return int(np.count_nonzero(seg_array) == 0)
    # treat axis 0 as frame axis
    nonzero_per_frame = np.array([np.count_nonzero(frame) for frame in seg_array])
    return int(np.sum(nonzero_per_frame == 0))


def analyze_patient(pid: str, files: dict):
    row = {
        "patient_id": pid,
        "has_ultrasound": "ultrasound" in files,
        "has_segmentation": "segmentation" in files,
    }

    us_arr = seg_arr = None

    if row["has_ultrasound"]:
        us_arr = load_array(files["ultrasound"])
    if row["has_segmentation"]:
        seg_arr = load_array(files["segmentation"])

    # --- Ultrasound stats ---
    if us_arr is not None:
        row["us_shape"] = us_arr.shape
        row["us_num_images"] = us_arr.shape[0] if us_arr.ndim >= 3 else 1
        row["us_resolution"] = us_arr.shape[1:3] if us_arr.ndim >= 3 else us_arr.shape
        row["us_min"] = float(us_arr.min())
        row["us_max"] = float(us_arr.max())
        row["us_dtype"] = str(us_arr.dtype)
    else:
        row["us_shape"] = None
        row["us_num_images"] = None
        row["us_resolution"] = None
        row["us_min"] = None
        row["us_max"] = None
        row["us_dtype"] = None

    # --- Segmentation stats ---
    if seg_arr is not None:
        row["seg_shape"] = seg_arr.shape
        row["seg_num_images"] = seg_arr.shape[0] if seg_arr.ndim >= 3 else 1
        row["seg_resolution"] = seg_arr.shape[1:3] if seg_arr.ndim >= 3 else seg_arr.shape
        row["seg_min"] = float(seg_arr.min())
        row["seg_max"] = float(seg_arr.max())
        row["seg_dtype"] = str(seg_arr.dtype)
        row["num_empty_masks"] = count_empty_masks(seg_arr)
    else:
        row["seg_shape"] = None
        row["seg_num_images"] = None
        row["seg_resolution"] = None
        row["seg_min"] = None
        row["seg_max"] = None
        row["seg_dtype"] = None
        row["num_empty_masks"] = None

    # --- Cross-checks / flags ---
    flags = []
    if not row["has_ultrasound"]:
        flags.append("MISSING_ULTRASOUND")
    if not row["has_segmentation"]:
        flags.append("MISSING_SEGMENTATION")
    if row["has_ultrasound"] and row["has_segmentation"]:
        if row["us_num_images"] != row["seg_num_images"]:
            flags.append(
                f"IMAGE_COUNT_MISMATCH (us={row['us_num_images']}, "
                f"seg={row['seg_num_images']})"
            )
        if row["us_resolution"] != row["seg_resolution"]:
            flags.append(
                f"RESOLUTION_MISMATCH (us={row['us_resolution']}, "
                f"seg={row['seg_resolution']})"
            )

    row["flags"] = "; ".join(flags) if flags else ""
    return row


def main():
    if not FOLDER.exists():
        print(f"Folder not found: {FOLDER.resolve()}")
        return

    patients = find_patient_files(FOLDER)
    print(f"Found {len(patients)} patient ID(s) in {FOLDER}\n")

    rows = []
    for pid, files in patients.items():
        print(f"Processing patient {pid} ...")
        rows.append(analyze_patient(pid, files))

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=== SUMMARY ===")
    print(df.to_string(index=False))

    flagged = df[df["flags"] != ""]
    print(f"\n{len(flagged)} patient(s) with issues flagged:")
    if not flagged.empty:
        print(flagged[["patient_id", "flags"]].to_string(index=False))
    else:
        print("  None — all patients passed the sanity checks.")

    total_empty = df["num_empty_masks"].dropna().sum()
    print(f"\nTotal empty masks across all patients: {int(total_empty)}")

    # Check resolution consistency across the whole dataset
    all_res = pd.concat([df["us_resolution"], df["seg_resolution"]]).dropna().unique()
    if len(all_res) > 1:
        print(f"\n[WARNING] Multiple distinct resolutions found across dataset: {list(all_res)}")
    elif len(all_res) == 1:
        print(f"\nAll images share a consistent resolution: {all_res[0]}")

    print(f"\nFull results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()