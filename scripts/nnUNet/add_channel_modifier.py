"""
Create symlinks for all image PNGs into a new folder, renaming each file
to insert a suffix (default "_0000") before the extension.

Example:
    LN003_frame00000.png  ->  LN003_frame00000_0000.png

This is commonly needed for frameworks like nnU-Net, which expect a
zero-padded channel index suffix on every input image filename.

The files themselves are NOT copied or moved -- only symlinks are
created in DEST_FOLDER, each pointing back at the original file in
SOURCE_FOLDER. This keeps disk usage minimal.
"""

from pathlib import Path

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images")
DEST_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/nnUNet_raw/Dataset067_LumpNav/imagesTr")
SUFFIX = "_0000"
# -----------------------------------------------------------------------


def main():
    if not SOURCE_FOLDER.exists():
        print(f"Source folder not found: {SOURCE_FOLDER.resolve()}")
        return

    DEST_FOLDER.mkdir(parents=True, exist_ok=True)

    png_files = sorted(SOURCE_FOLDER.glob("*.png"))
    print(f"Found {len(png_files)} PNG file(s) in {SOURCE_FOLDER}\n")

    n_created = n_skipped = n_error = 0

    for src in png_files:
        new_name = f"{src.stem}{SUFFIX}{src.suffix}"
        dst = DEST_FOLDER / new_name

        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            n_created += 1
        except Exception as e:
            print(f"  [error] Failed to symlink {src.name} -> {new_name}: {e}")
            n_error += 1

    print(f"Created {n_created} symlink(s) in {DEST_FOLDER.resolve()}")
    if n_error:
        print(f"{n_error} file(s) failed -- see errors above")


if __name__ == "__main__":
    main()