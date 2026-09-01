"""
Crop the larger-resolution (616x820) ultrasound images/masks so their
black border matches the smaller resolution (615x525), rather than
stripping borders entirely.

Based on the measured border widths:

    Side    | small (615x525) | large (616x820) | crop needed
    --------|------------------|------------------|-------------
    top     | X                | X (same)         | 0
    bottom  | X                | X + 1            | 1 px off bottom
    left    | 32               | 177              | 145 px off left
    right   | 26               | 176              | 150 px off right

    615 = 616 - 1   (top unchanged, 1 px removed from bottom)
    525 = 820 - 145 - 150   (145 px removed from left, 150 px from right)

This assumes the SAME crop amounts apply to every large-resolution
image (i.e. the border offset is a fixed, consistent padding rather
than varying per frame -- confirm this by checking the std values from
the earlier border-detection script; if std was non-zero, this fixed
crop may not be exactly right for every frame and per-frame detection
would be safer).

Both the images/ and masks/ folders are processed so that image/mask
pairs stay aligned in size after cropping. Cropped results are written
to a NEW output folder -- originals are never overwritten in place.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNG")
OUTPUT_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched")
SUBDIRS = ["images", "masks"]

SMALL_RES = (615, 525)   # (height, width) -- left as-is
LARGE_RES = (616, 820)   # (height, width) -- will be cropped

# Crop amounts derived from measured borders (see docstring above)
CROP_TOP = 0
CROP_BOTTOM = 1
CROP_LEFT = 145
CROP_RIGHT = 150
# -----------------------------------------------------------------------


def crop_large_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cropped = img[CROP_TOP: h - CROP_BOTTOM, CROP_LEFT: w - CROP_RIGHT]
    return cropped


def process_folder(subdir: str):
    src_dir = SOURCE_FOLDER / subdir
    out_dir = OUTPUT_FOLDER / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(f"  [skip] {src_dir} does not exist")
        return

    png_files = sorted(src_dir.glob("*.png"))
    print(f"\nProcessing '{subdir}' ({len(png_files)} files)...")

    n_small = n_large_cropped = n_unexpected = n_error = 0

    for path in png_files:
        try:
            img = np.array(Image.open(path))
            h, w = img.shape[:2]

            if (h, w) == SMALL_RES:
                # Already the target resolution -- copy through unchanged
                Image.fromarray(img).save(out_dir / path.name)
                n_small += 1

            elif (h, w) == LARGE_RES:
                cropped = crop_large_image(img)
                if cropped.shape[:2] != SMALL_RES:
                    print(
                        f"  [warning] {path.name}: cropped shape "
                        f"{cropped.shape[:2]} does not match expected "
                        f"{SMALL_RES} -- check crop constants"
                    )
                    n_unexpected += 1
                Image.fromarray(cropped).save(out_dir / path.name)
                n_large_cropped += 1

            else:
                print(
                    f"  [warning] {path.name}: unexpected resolution "
                    f"{(h, w)} (not {SMALL_RES} or {LARGE_RES}) -- copied "
                    f"through unchanged"
                )
                Image.fromarray(img).save(out_dir / path.name)
                n_unexpected += 1

        except Exception as e:
            print(f"  [error] Failed to process {path.name}: {e}")
            n_error += 1

    print(
        f"  done: {n_small} already correct size, {n_large_cropped} cropped, "
        f"{n_unexpected} unexpected resolution, {n_error} errors"
    )


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    for subdir in SUBDIRS:
        process_folder(subdir)

    print(f"\nCropped dataset written to: {OUTPUT_FOLDER.resolve()}")
    print(
        "All images/masks should now be "
        f"{SMALL_RES[0]}x{SMALL_RES[1]} (HxW). Re-run the earlier "
        "resolution/border-detection script on this folder to confirm."
    )


if __name__ == "__main__":
    main()