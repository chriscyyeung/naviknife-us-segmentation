"""
Convert mask PNGs to boolean masks, saved as PIL mode "1" (bilevel)
PNGs -- directly viewable as black/white AND loads back as a true
`bool` NumPy array, no separate 0/255 rescaling needed for viewing.

    pixel > THRESHOLD  ->  True  (foreground / tumor)
    pixel <= THRESHOLD ->  False (background)

Files are written to a NEW output folder; originals are untouched.

Note: this format is for visualization / general boolean-logic use.
Frameworks like nnU-Net expect integer label maps ({0,1} as uint8),
not boolean PNGs -- use the earlier binarization script for that
pipeline instead. This script is for a boolean/viewable representation,
not a drop-in replacement for nnU-Net's expected input format.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks_8bit")
DEST_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks")
THRESHOLD = 128     # pixel values above this become True (foreground)
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

    for src in png_files:
        try:
            mask = np.array(Image.open(src).convert("L"))
            bool_mask = mask > THRESHOLD

            if not bool_mask.any():
                empty_masks += 1

            # PIL auto-detects mode "1" (bilevel) for boolean arrays --
            # directly viewable as black/white, no scaling needed
            Image.fromarray(bool_mask).save(DEST_FOLDER / src.name)
            n_converted += 1

        except Exception as e:
            print(f"  [error] Failed to convert {src.name}: {e}")
            n_error += 1

    print(f"Converted {n_converted} mask(s) to boolean PNGs in {DEST_FOLDER.resolve()}")
    if n_error:
        print(f"{n_error} file(s) failed -- see errors above")
    print(f"Empty (all-background) masks after thresholding: {empty_masks}")


if __name__ == "__main__":
    main()