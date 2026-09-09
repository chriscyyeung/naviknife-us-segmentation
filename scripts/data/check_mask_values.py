"""
Scan the (pre-binarization) mask PNGs to find files containing pixel
values other than the expected 0/255, e.g. 127 -- which suggests
antialiasing, JPEG-like compression artifacts (PNG is lossless, so this
would more likely come from resizing/cropping with interpolation), or a
genuine third class.

For each mask file, reports:
    - all distinct pixel values found
    - how many pixels have each "unexpected" value (anything not in
      EXPECTED_VALUES)
    - where those pixels are located (bounding box), which helps tell
      whether they're scattered noise (e.g. isolated antialiased edge
      pixels) vs. a solid region (e.g. a real third class)

Outputs a CSV listing only the flagged files with their unexpected-value
details, plus a console summary.
"""

from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks")
EXPECTED_VALUES = {0, 255}
OUTPUT_CSV = "unexpected_mask_values.csv"
# -----------------------------------------------------------------------


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    png_files = sorted(SOURCE_FOLDER.glob("*.png"))
    print(f"Scanning {len(png_files)} mask file(s) in {SOURCE_FOLDER}\n")

    flagged_rows = []
    global_value_counts = Counter()

    for i, path in enumerate(png_files, 1):
        if i % 500 == 0:
            print(f"  ...scanned {i}/{len(png_files)}")

        try:
            mask = np.array(Image.open(path).convert("L"))
        except Exception as e:
            print(f"  [error] Failed to read {path.name}: {e}")
            continue

        values, counts = np.unique(mask, return_counts=True)
        for v, c in zip(values, counts):
            global_value_counts[int(v)] += int(c)

        unexpected = {int(v): int(c) for v, c in zip(values, counts) if int(v) not in EXPECTED_VALUES}

        if unexpected:
            # For the dominant unexpected value, find where those pixels are
            details = []
            for val, count in sorted(unexpected.items(), key=lambda x: -x[1]):
                ys, xs = np.where(mask == val)
                bbox = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
                details.append(f"value={val} count={count} bbox_rows={bbox[0]}-{bbox[1]} bbox_cols={bbox[2]}-{bbox[3]}")

            flagged_rows.append({
                "file": path.name,
                "unexpected_values": sorted(unexpected.keys()),
                "unexpected_pixel_counts": unexpected,
                "details": "; ".join(details),
            })

    print(f"\n=== GLOBAL PIXEL VALUE COUNTS ACROSS ALL MASKS ===")
    for val, count in sorted(global_value_counts.items()):
        flag = "  <-- unexpected" if val not in EXPECTED_VALUES else ""
        print(f"  value {val}: {count} pixels{flag}")

    print(f"\n=== FILES WITH UNEXPECTED PIXEL VALUES: {len(flagged_rows)} ===")
    for row in flagged_rows:
        print(f"  {row['file']}: {row['details']}")

    if flagged_rows:
        df = pd.DataFrame(flagged_rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFlagged file details saved to: {OUTPUT_CSV}")
    else:
        print("\nNo unexpected pixel values found -- all masks are strictly 0/255.")


if __name__ == "__main__":
    main()