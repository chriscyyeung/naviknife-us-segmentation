"""
Convert segmentation mask PNGs into nnU-Net-style binary label masks.

nnU-Net expects label images where each pixel holds an integer CLASS
INDEX (0 = background, 1 = class 1, 2 = class 2, ...), not a visually
scaled image. The masks exported earlier were saved as 8-bit PNGs with
values in {0, 255} (0 = background, 255 = tumor, for visualization).
This script converts those into proper label masks with values in
{0, 1}:

    pixel > THRESHOLD  ->  1   (tumor / foreground)
    pixel <= THRESHOLD ->  0   (background)

Output PNGs will look almost solid black when viewed normally, since
pixel value 1 is indistinguishable from 0 to the human eye -- that's
expected and correct for nnU-Net label maps (only the numeric values
matter, not visual appearance).

Files are written to a NEW output folder; originals are untouched.

Note: Source files were changed to bool, so this script is now outdated.
Conversion should now only require converting uint8 without thresholding 
(implemented but not tested).
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks")
DEST_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/nnUNet_raw/Dataset067_LumpNav/labelsTr")
# -----------------------------------------------------------------------


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    DEST_FOLDER.mkdir(parents=True, exist_ok=True)

    png_files = sorted(SOURCE_FOLDER.glob("*.png"))
    print(f"Found {len(png_files)} PNG file(s) in {SOURCE_FOLDER}\n")

    n_converted = n_error = 0
    empty_masks = 0
    unique_values_seen = set()

    for src in png_files:
        try:
            mask = np.array(Image.open(src))
            unique_values_seen.update(np.unique(mask).tolist())

            binary_mask = mask.astype(np.uint8)

            if binary_mask.max() == 0:
                empty_masks += 1

            # save as 8-bit single-channel PNG with values {0,1}
            Image.fromarray(binary_mask, mode="L").save(DEST_FOLDER / src.name)
            n_converted += 1

        except Exception as e:
            print(f"  [error] Failed to convert {src.name}: {e}")
            n_error += 1

    print(f"Converted {n_converted} mask(s) to binary labels in {DEST_FOLDER.resolve()}")
    if n_error:
        print(f"{n_error} file(s) failed -- see errors above")
    print(f"Empty (all-background) masks after thresholding: {empty_masks}")
    print(f"Distinct raw pixel values seen across all source masks: {sorted(unique_values_seen)}")

    if len(unique_values_seen) > 3:
        print(
            "\n[NOTE] More than 2-3 distinct pixel values were found in the "
            "source masks. If these represent more than one tumor class "
            "(rather than antialiasing/compression noise), a single "
            "threshold will collapse them into one class incorrectly -- "
            "let me know the class values and I can switch this to a "
            "proper multi-class mapping instead of a binary threshold."
        )


if __name__ == "__main__":
    main()