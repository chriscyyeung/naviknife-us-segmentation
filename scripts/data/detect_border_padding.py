"""
Detect black-border padding in exported ultrasound PNGs and compare it
across the two observed resolutions (615x525 and 616x820).

Approach
--------
For each image, a row/column is considered part of the black border if
EVERY pixel in it is at or below a small intensity threshold (to allow
for compression noise / near-black pixels rather than requiring exact
0). Starting from each edge, we count consecutive all-dark rows/columns
inward until we hit the first row/column with real content. This gives
four border widths per image: top, bottom, left, right.

We then group images by their (height, width) resolution and report,
per group:
    - mean / median / std / min / max border width on each side
    - the resulting "content resolution" (image size minus borders),
      which is what you actually want to compare between the two
      resolution groups to confirm they contain the same underlying
      content just padded differently

Results are printed and saved to CSV (one row per image) plus a
per-resolution summary CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ---- CONFIG ----------------------------------------------------------
IMAGES_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images")
DARK_THRESHOLD = 0          # pixel intensity <= this counts as "black"
PER_IMAGE_CSV = "border_detection_per_image.csv"
SUMMARY_CSV = "border_detection_summary.csv"
# -----------------------------------------------------------------------


def detect_borders(img: np.ndarray, threshold: int = DARK_THRESHOLD):
    """Return (top, bottom, left, right) border widths in pixels.

    A row/column counts as border only if ALL its pixels are <= threshold,
    so real (non-uniform) image content stops the count immediately.
    """
    dark_rows = np.all(img <= threshold, axis=1)
    dark_cols = np.all(img <= threshold, axis=0)

    top = int(np.argmax(~dark_rows)) if not dark_rows.all() else img.shape[0]
    bottom = int(np.argmax(~dark_rows[::-1])) if not dark_rows.all() else img.shape[0]
    left = int(np.argmax(~dark_cols)) if not dark_cols.all() else img.shape[1]
    right = int(np.argmax(~dark_cols[::-1])) if not dark_cols.all() else img.shape[1]

    return top, bottom, left, right


def analyze_image(path: Path) -> dict:
    img = np.array(Image.open(path).convert("L"))
    h, w = img.shape
    top, bottom, left, right = detect_borders(img)
    content_h = h - top - bottom
    content_w = w - left - right
    return {
        "file": path.name,
        "height": h,
        "width": w,
        "resolution": f"{h}x{w}",
        "border_top": top,
        "border_bottom": bottom,
        "border_left": left,
        "border_right": right,
        "content_height": content_h,
        "content_width": content_w,
    }


def main():
    if not IMAGES_FOLDER.exists():
        print(f"Images folder not found: {IMAGES_FOLDER.resolve()}")
        return

    png_files = sorted(IMAGES_FOLDER.glob("*.png"))
    print(f"Found {len(png_files)} PNG file(s) in {IMAGES_FOLDER}\n")

    rows = []
    for i, path in enumerate(png_files, 1):
        if i % 200 == 0:
            print(f"  ...processed {i}/{len(png_files)}")
        try:
            rows.append(analyze_image(path))
        except Exception as e:
            print(f"  [error] Failed to process {path.name}: {e}")

    if not rows:
        print("No images successfully processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(PER_IMAGE_CSV, index=False)

    print("\n=== RESOLUTIONS FOUND ===")
    print(df["resolution"].value_counts().to_string())

    print("\n=== PER-RESOLUTION BORDER SUMMARY ===")
    summary_rows = []
    for res, group in df.groupby("resolution"):
        summary = {
            "resolution": res,
            "num_images": len(group),
        }
        for side in ["border_top", "border_bottom", "border_left", "border_right"]:
            summary[f"{side}_mean"] = round(group[side].mean(), 2)
            summary[f"{side}_median"] = group[side].median()
            summary[f"{side}_std"] = round(group[side].std(), 2)
            summary[f"{side}_min"] = group[side].min()
            summary[f"{side}_max"] = group[side].max()
        summary["content_height_mean"] = round(group["content_height"].mean(), 2)
        summary["content_width_mean"] = round(group["content_width"].mean(), 2)
        summary["content_height_std"] = round(group["content_height"].std(), 2)
        summary["content_width_std"] = round(group["content_width"].std(), 2)
        summary_rows.append(summary)

        print(f"\nResolution {res} (n={len(group)}):")
        print(f"  top    border: mean={summary['border_top_mean']:>6}  "
              f"median={summary['border_top_median']:>6}  "
              f"std={summary['border_top_std']:>6}  "
              f"range=[{summary['border_top_min']}, {summary['border_top_max']}]")
        print(f"  bottom border: mean={summary['border_bottom_mean']:>6}  "
              f"median={summary['border_bottom_median']:>6}  "
              f"std={summary['border_bottom_std']:>6}  "
              f"range=[{summary['border_bottom_min']}, {summary['border_bottom_max']}]")
        print(f"  left   border: mean={summary['border_left_mean']:>6}  "
              f"median={summary['border_left_median']:>6}  "
              f"std={summary['border_left_std']:>6}  "
              f"range=[{summary['border_left_min']}, {summary['border_left_max']}]")
        print(f"  right  border: mean={summary['border_right_mean']:>6}  "
              f"median={summary['border_right_median']:>6}  "
              f"std={summary['border_right_std']:>6}  "
              f"range=[{summary['border_right_min']}, {summary['border_right_max']}]")
        print(f"  --> content region (border stripped): "
              f"{summary['content_height_mean']} x {summary['content_width_mean']} "
              f"(std: {summary['content_height_std']} x {summary['content_width_std']})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # --- Direct comparison across resolutions, if exactly two groups ---
    if len(summary_df) == 2:
        print("\n=== CROSS-RESOLUTION COMPARISON ===")
        r1, r2 = summary_df.iloc[0], summary_df.iloc[1]
        h_diff = abs(r1["content_height_mean"] - r2["content_height_mean"])
        w_diff = abs(r1["content_width_mean"] - r2["content_width_mean"])
        print(f"Content height difference between groups: {h_diff:.2f} px")
        print(f"Content width difference between groups:  {w_diff:.2f} px")
        if h_diff < 3 and w_diff < 3:
            print(
                "-> Content regions are essentially the same size across both "
                "resolutions, supporting the hypothesis that the difference is "
                "purely extra black border padding."
            )
        else:
            print(
                "-> Content regions differ meaningfully in size between "
                "resolutions -- this is NOT simply a border/padding "
                "difference; the underlying image content itself differs "
                "in scale."
            )

    print(f"\nPer-image results saved to: {PER_IMAGE_CSV}")
    print(f"Per-resolution summary saved to: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()