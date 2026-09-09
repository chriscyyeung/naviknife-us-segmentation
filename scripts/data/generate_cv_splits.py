"""
Generate a K-fold (default 5) cross-validation split of the cropped
LumpNav dataset, grouped by patient (every frame from one patient stays
in a single fold -- never split across folds).

Folds are balanced by IMAGE count (not just patient count), using a
greedy load-balancing assignment: patients are processed largest-first
and each is assigned to whichever fold currently has the fewest images,
so fold sizes stay as even as possible despite patients having very
different frame counts.

For each fold i (0..K-1):
    - fold_i's assigned patients become that fold's VALIDATION set
    - all remaining patients become that fold's TRAIN set

Output structure:
    <OUTPUT_FOLDER>/
        fold_0/
            train/{images,masks}/
            val/{images,masks}/
        fold_1/
            ...
        ...
        cv_manifest.csv   -- patient_id, fold, num_images

By default files are SYMLINKED rather than copied. With K=5, each
patient's images appear in 4 of 5 training sets, so copying real files
would multiply disk usage ~4-5x; symlinks avoid that. Set COPY_MODE =
"copy" if you need real file copies (e.g. transferring the split
elsewhere).
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---- CONFIG ----------------------------------------------------------
SOURCE_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched")
OUTPUT_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatchedCV")
SUBDIRS = ["images", "masks"]

N_FOLDS = 5
COPY_MODE = "symlink"     # "copy" or "symlink" -- symlink strongly recommended for k-fold
MANIFEST_CSV = "cv_split_manifest.csv"
# -----------------------------------------------------------------------

FNAME_RE = re.compile(r"^(?P<pid>LN\d+)_frame\d+\.png$", re.IGNORECASE)


def get_patient_image_counts(images_dir: Path):
    patient_files = defaultdict(list)
    for f in sorted(images_dir.glob("*.png")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  [skip] Filename doesn't match expected pattern: {f.name}")
            continue
        patient_files[m.group("pid")].append(f.name)
    return patient_files


def balanced_kfold_assignment(patient_files: dict, k: int):
    """Greedily assign whole patients to k folds, balancing total image
    count per fold. Returns {patient_id: fold_index}."""
    patients = sorted(patient_files.items(), key=lambda kv: len(kv[1]), reverse=True)
    fold_totals = [0] * k
    assignment = {}
    for pid, files in patients:
        # assign to the fold with the smallest current total
        target_fold = min(range(k), key=lambda i: fold_totals[i])
        assignment[pid] = target_fold
        fold_totals[target_fold] += len(files)
    return assignment, fold_totals


def copy_or_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if COPY_MODE == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main():
    images_dir = SOURCE_FOLDER / "images"
    if not images_dir.exists():
        print(f"Images folder not found: {images_dir.resolve()}")
        return

    patient_files = get_patient_image_counts(images_dir)
    n_patients = len(patient_files)
    n_images = sum(len(v) for v in patient_files.values())
    print(f"Found {n_patients} patient(s), {n_images} image(s) total in {images_dir}\n")

    if n_patients < N_FOLDS:
        print(f"[error] Only {n_patients} patients found, cannot make {N_FOLDS} folds.")
        return

    assignment, fold_totals = balanced_kfold_assignment(patient_files, N_FOLDS)

    print(f"=== {N_FOLDS}-FOLD ASSIGNMENT (balanced by image count) ===")
    fold_patients = defaultdict(list)
    for pid, fold in assignment.items():
        fold_patients[fold].append(pid)
    for fold in range(N_FOLDS):
        pids = sorted(fold_patients[fold])
        print(f"Fold {fold}: {len(pids)} patients, {fold_totals[fold]} images "
              f"({fold_totals[fold] / n_images:.1%}) -- {pids}")

    # --- Build folder structure for each fold ---
    manifest_rows = []
    for pid, fold in assignment.items():
        manifest_rows.append({
            "patient_id": pid,
            "fold": fold,
            "num_images": len(patient_files[pid]),
        })
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["fold", "patient_id"])
    manifest_df.to_csv(MANIFEST_CSV, index=False)
    print(f"\nManifest saved to: {MANIFEST_CSV}")

    for val_fold in range(N_FOLDS):
        print(f"\nBuilding fold_{val_fold} ({COPY_MODE})...")
        val_patients = set(fold_patients[val_fold])
        train_patients = set(patient_files.keys()) - val_patients

        for subdir in SUBDIRS:
            src_dir = SOURCE_FOLDER / subdir
            if not src_dir.exists():
                print(f"  [skip] {src_dir} does not exist, skipping this subfolder")
                continue

            for pid in train_patients:
                for fname in patient_files[pid]:
                    src = src_dir / fname
                    if not src.exists():
                        continue
                    dst = OUTPUT_FOLDER / f"fold_{val_fold}" / "train" / subdir / fname
                    copy_or_link(src, dst)

            for pid in val_patients:
                for fname in patient_files[pid]:
                    src = src_dir / fname
                    if not src.exists():
                        continue
                    dst = OUTPUT_FOLDER / f"fold_{val_fold}" / "val" / subdir / fname
                    copy_or_link(src, dst)

    print(f"\n{N_FOLDS}-fold CV dataset written to: {OUTPUT_FOLDER.resolve()}")
    print(
        "Each fold_i/train and fold_i/val folder contains images/ and masks/ "
        "subfolders. Train a model on fold_i/train and evaluate on fold_i/val "
        "for each i, then average metrics across folds."
    )


if __name__ == "__main__":
    main()