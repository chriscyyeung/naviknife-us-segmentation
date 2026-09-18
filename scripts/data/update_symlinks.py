"""
Update existing symlinks (e.g. in your CV-split or train/test-split
folders) so they point at the new boolean masks instead of whatever
mask version they currently reference.

Walks SEARCH_ROOT recursively, and for every symlink found inside a
directory named MASKS_DIR_NAME (default "masks"), re-targets it to the
file with the same filename in NEW_MASKS_FOLDER -- regardless of what
it currently points to. Files outside a "masks"-named directory (e.g.
symlinks in "images/") are left untouched, so this is safe to run even
if images and masks share filenames.

Only symlinks are touched -- real files (non-symlinks) are reported but
skipped, since this script is specifically for repointing links, not
replacing actual file content.
"""

from pathlib import Path

# ---- CONFIG ----------------------------------------------------------
SEARCH_ROOT = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV")          # tree to scan for symlinks
NEW_MASKS_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/masks")
MASKS_DIR_NAME = "masks"    # only symlinks inside a folder with this name are touched
# -----------------------------------------------------------------------


def main():
    if not SEARCH_ROOT.exists():
        print(f"Search root not found: {SEARCH_ROOT.resolve()}")
        return
    if not NEW_MASKS_FOLDER.exists():
        print(f"New masks folder not found: {NEW_MASKS_FOLDER.resolve()}")
        return

    n_updated = n_skipped_not_symlink = n_missing_target = n_already_correct = 0

    # Walk every directory named MASKS_DIR_NAME under SEARCH_ROOT
    for masks_dir in SEARCH_ROOT.rglob(MASKS_DIR_NAME):
        if not masks_dir.is_dir():
            continue

        print(f"Processing {masks_dir} ...")

        for entry in sorted(masks_dir.iterdir()):
            if not entry.is_symlink():
                print(f"  [skip] {entry.name}: not a symlink, leaving as-is")
                n_skipped_not_symlink += 1
                continue

            new_target = (NEW_MASKS_FOLDER / entry.name).resolve()

            if not new_target.exists():
                print(f"  [warning] {entry.name}: no matching file in "
                      f"{NEW_MASKS_FOLDER}, leaving symlink unchanged")
                n_missing_target += 1
                continue

            current_target = entry.resolve()
            if current_target == new_target:
                n_already_correct += 1
                continue

            entry.unlink()
            entry.symlink_to(new_target)
            n_updated += 1

    print("\n=== SUMMARY ===")
    print(f"Symlinks updated: {n_updated}")
    print(f"Already pointing at the correct target: {n_already_correct}")
    print(f"Non-symlink files skipped: {n_skipped_not_symlink}")
    print(f"No matching new mask found: {n_missing_target}")


if __name__ == "__main__":
    main()