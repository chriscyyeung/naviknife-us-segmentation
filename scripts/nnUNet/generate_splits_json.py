"""
Generate an nnU-Net splits_final.json from the patient-level cross-
validation assignment in cv_split_manifest.csv.

nnU-Net's splits_final.json format is a list of dicts, one per fold:

    [
        {"train": ["case_id_1", "case_id_2", ...], "val": ["case_id_9", ...]},
        {"train": [...], "val": [...]},
        ...
    ]

Each "case identifier" is the image filename WITHOUT extension and
WITHOUT the channel suffix (e.g. "LN003_frame00000_0000.png" -> case id
"LN003_frame00000"). This script reads the actual filenames from the
(pre-channel-suffix) cropped images folder, groups them by patient, and
uses the fold assignment from cv_split_manifest.csv to build the final
train/val lists for each fold.
"""

import re
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---- CONFIG ----------------------------------------------------------
IMAGES_FOLDER = Path("/home/cyeung/projects/aip-medilab/cyeung/LumpNavPNGResMatched/images")   # pre-channel-suffix images
MANIFEST_CSV = Path("/home/cyeung/projects/aip-medilab/cyeung/naviknife-us-segmentation/cv_split_manifest.csv")
OUTPUT_JSON = Path("/home/cyeung/projects/aip-medilab/cyeung/nnUNet_preprocessed/Dataset067_LumpNav/splits_final.json")
N_FOLDS = 5
# -----------------------------------------------------------------------

FNAME_RE = re.compile(r"^(?P<pid>LN\d+)_frame\d+\.png$", re.IGNORECASE)


def get_patient_case_ids(images_dir: Path):
    """Return {patient_id: [case_id, case_id, ...]} where case_id is the
    filename stem (no extension, no channel suffix)."""
    patient_cases = defaultdict(list)
    for f in sorted(images_dir.glob("*.png")):
        m = FNAME_RE.match(f.name)
        if not m:
            print(f"  [skip] Filename doesn't match expected pattern: {f.name}")
            continue
        patient_cases[m.group("pid")].append(f.stem)  # e.g. "LN003_frame00000"
    return patient_cases


def main():
    if not IMAGES_FOLDER.exists():
        print(f"Images folder not found: {IMAGES_FOLDER.resolve()}")
        return
    if not MANIFEST_CSV.exists():
        print(f"Manifest CSV not found: {MANIFEST_CSV.resolve()}")
        return

    patient_cases = get_patient_case_ids(IMAGES_FOLDER)
    manifest_df = pd.read_csv(MANIFEST_CSV)

    missing_in_manifest = set(patient_cases.keys()) - set(manifest_df["patient_id"])
    missing_in_images = set(manifest_df["patient_id"]) - set(patient_cases.keys())
    if missing_in_manifest:
        print(f"[warning] Patients found in images folder but not in manifest: "
              f"{sorted(missing_in_manifest)}")
    if missing_in_images:
        print(f"[warning] Patients found in manifest but not in images folder: "
              f"{sorted(missing_in_images)}")

    patient_to_fold = dict(zip(manifest_df["patient_id"], manifest_df["fold"]))

    splits = []
    for fold in range(N_FOLDS):
        val_patients = [p for p, f in patient_to_fold.items() if f == fold]
        train_patients = [p for p, f in patient_to_fold.items() if f != fold]

        val_cases = sorted(
            case_id for p in val_patients for case_id in patient_cases.get(p, [])
        )
        train_cases = sorted(
            case_id for p in train_patients for case_id in patient_cases.get(p, [])
        )

        splits.append({"train": train_cases, "val": val_cases})

        print(f"Fold {fold}: {len(train_patients)} train patients "
              f"({len(train_cases)} cases), {len(val_patients)} val patients "
              f"({len(val_cases)} cases)")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nsplits_final.json written to: {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()