"""
Fix LN015 mask files: pixels with value 127 (a second, mislabeled
segmentation representing the same class) are converted to 255 so they
merge into the single tumor class, matching all other masks.

Only files matching the LN015 patient prefix are touched. A backup copy
of each modified file is saved before overwriting, in case you need to
revert.

Run this BEFORE the binary-conversion script, so the corrected 0/255
masks feed cleanly into thresholding.
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks")
PATIENT_PREFIX = "LN015"
VALUE_TO_FIX = 127
FIX_TO_VALUE = 255
BACKUP_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks_backup_before_fix")
# -----------------------------------------------------------------------


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    target_files = sorted(SOURCE_FOLDER.glob(f"{PATIENT_PREFIX}_*.png"))
    print(f"Found {len(target_files)} mask file(s) for {PATIENT_PREFIX}\n")

    BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)

    n_fixed = n_unaffected = n_error = 0

    for path in target_files:
        try:
            mask = np.array(Image.open(path).convert("L"))

            if not np.any(mask == VALUE_TO_FIX):
                n_unaffected += 1
                continue

            n_pixels = int(np.sum(mask == VALUE_TO_FIX))

            # Back up original before modifying
            shutil.copy2(path, BACKUP_FOLDER / path.name)

            fixed_mask = mask.copy()
            fixed_mask[fixed_mask == VALUE_TO_FIX] = FIX_TO_VALUE

            Image.fromarray(fixed_mask, mode="L").save(path)

            print(f"  fixed {path.name}: {n_pixels} pixel(s) changed from "
                  f"{VALUE_TO_FIX} -> {FIX_TO_VALUE}")
            n_fixed += 1

        except Exception as e:
            print(f"  [error] Failed to process {path.name}: {e}")
            n_error += 1

    print(f"\nFixed {n_fixed} file(s), {n_unaffected} unaffected "
          f"(no value {VALUE_TO_FIX} present), {n_error} error(s)")
    print(f"Backups of modified files saved to: {BACKUP_FOLDER.resolve()}")


if __name__ == "__main__":
    main()