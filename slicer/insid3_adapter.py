"""
TorchSequenceSegmentation adapter for INSID3 with a LoRA-adapted DINOv3 backbone
(train_lora_lumpnav.py in the INSID3 fork), ensembled over the 5 CV folds.

INSID3 is in-context: every frame is segmented against a fixed reference set of
LumpNav images + tumour masks. The references are encoded once in load_model()
and reused for every frame, so each frame only costs one backbone forward pass
per fold (plus INSID3's clustering).

Model folder layout (the folder selected in the Slicer module):

    <model_dir>/
        insid3_adapter.py                     <- this file
        fold_0/dinov3_vitl16_lumpnav_lora_fold0-lora0000.pth
        ...                                   (any depth; found by file name)
        fold_4/dinov3_vitl16_lumpnav_lora_fold4-lora0004.pth
        references/images/LNxxx_frameNNNNN.png
        references/masks/LNxxx_frameNNNNN.png  (same file names, any non-zero = tumour)

The merged backbones are the files train_insid3_lora.sh writes to
external/INSID3/output/insid3_lora/fold_<k>/train/. Build references/ with
make_insid3_references.py (one non-empty frame per patient, 5-shot by default).

Requirements in Slicer's Python: torch, torchvision, einops, scikit-learn,
pycocotools (installed on first use if missing), plus a checkout of the INSID3
fork (`git submodule update --init external/INSID3`, branch fork-changes).
DINOv3's model code comes from torch.hub (needs internet once, then cached),
or from a local clone of facebookresearch/dinov3 via DINOV3_REPO_DIR.

Output: the fraction of fold models that mark each pixel as tumour (0, 0.2, ...,
1.0 with 5 folds), so the module's 0.5 threshold is a majority vote.
"""

import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---- Configuration -- adjust these for your environment/checkpoint layout ----
INSID3_REPO_DIR = os.environ.get(
    "INSID3_REPO_DIR",
    r"C:\Users\chris\Repos\naviknife-us-segmentation\external\INSID3",
)
# Optional local clone of https://github.com/facebookresearch/dinov3 (offline use).
# Empty -> torch.hub.load("facebookresearch/dinov3", ...) from GitHub / the hub cache.
DINOV3_REPO_DIR = os.environ.get("DINOV3_REPO_DIR", "")
BACKBONE_GLOB = "dinov3_vit*16_lumpnav_lora_fold*-*.pth"
REFERENCE_DIR_NAME = "references"
FOLDS = None            # None = every fold found; e.g. [0] to run one fold (5x faster)

# INSID3 settings: keep identical to the LumpNav evaluation (opts.py defaults,
# --image-size 1024 as in train_insid3_lora.sh). The LoRA was trained at 1024.
IMAGE_SIZE = 1024
SVD_COMPONENTS = 500
TAU = 0.6
MERGE_THRESHOLD = 0.2

_HUB_NAMES = {"vits16": "dinov3_vits16", "vitb16": "dinov3_vitb16", "vitl16": "dinov3_vitl16"}
_FOLD_RE = re.compile(r"dinov3_(vit[slb]16)_lumpnav_lora_fold(\d+)-")


def _install_dependencies():
    missing = []
    for module, package in (("einops", "einops"), ("sklearn", "scikit-learn"),
                            ("pycocotools", "pycocotools"), ("torchvision", "torchvision"),
                            ("torchmetrics", "torchmetrics"), ("termcolor", "termcolor")):
        # torchmetrics and termcolor are imported by DINOv3's hubconf, not by INSID3 itself
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        import slicer
        slicer.util.pip_install(" ".join(missing))


def _import_insid3():
    """Import INSID3 from its repo. Its top-level packages are called `models` and
    `utils`, the same as OpenUS's, so drop any cached copies that came from another
    repo (e.g. OpenUS loaded earlier in the same Slicer session) before importing."""
    if not os.path.isfile(os.path.join(INSID3_REPO_DIR, "models", "insid3.py")):
        raise FileNotFoundError(
            f"INSID3 code not found in {INSID3_REPO_DIR}. Run "
            "`git submodule update --init external/INSID3` in naviknife-us-segmentation, "
            "or set INSID3_REPO_DIR to your INSID3 fork checkout (branch fork-changes)."
        )
    repo = os.path.normcase(os.path.abspath(INSID3_REPO_DIR))
    for name in list(sys.modules):
        if name.split(".")[0] in ("models", "utils"):
            path = getattr(sys.modules[name], "__file__", None) or ""
            if not os.path.normcase(os.path.abspath(path)).startswith(repo):
                del sys.modules[name]
    if INSID3_REPO_DIR in sys.path:
        sys.path.remove(INSID3_REPO_DIR)
    sys.path.insert(0, INSID3_REPO_DIR)

    from models.insid3 import INSID3
    from utils.data import build_transform, downsample_mask
    from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
    from utils.refinement import upsample_mask
    return {
        "INSID3": INSID3, "build_transform": build_transform, "downsample_mask": downsample_mask,
        "agglomerative_clustering": agglomerative_clustering,
        "compute_cluster_prototypes": compute_cluster_prototypes, "upsample_mask": upsample_mask,
    }


def _find_backbones(model_dir):
    found = {}
    for path in glob.glob(os.path.join(model_dir, "**", BACKBONE_GLOB), recursive=True):
        match = _FOLD_RE.search(os.path.basename(path))
        if not match:
            continue
        arch, fold = match.group(1), int(match.group(2))
        if fold in found:
            raise ValueError(f"Two backbones for fold {fold} under {model_dir}: "
                             f"{found[fold][1]} and {path}. Remove one so folds are unambiguous.")
        found[fold] = (arch, path)
    if FOLDS is not None:
        found = {k: v for k, v in found.items() if k in FOLDS}
    return dict(sorted(found.items()))


def _build_encoder(arch, weights_path):
    hub_name = _HUB_NAMES[arch]
    if DINOV3_REPO_DIR:
        encoder = torch.hub.load(DINOV3_REPO_DIR, hub_name, source="local", pretrained=False)
    else:
        encoder = torch.hub.load("facebookresearch/dinov3", hub_name, pretrained=False)
    # Same as INSID3's models._build_encoder: load the merged LoRA checkpoint strictly from disk
    encoder.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return encoder


def _read_rgb(path):
    return Image.open(path).convert("RGB")


def _load_references(model_dir):
    ref_dir = os.path.join(model_dir, REFERENCE_DIR_NAME)
    image_paths = sorted(glob.glob(os.path.join(ref_dir, "images", "*.png")))
    pairs = []
    for image_path in image_paths:
        mask_path = os.path.join(ref_dir, "masks", os.path.basename(image_path))
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Reference image {image_path} has no mask at {mask_path}")
        mask = np.array(Image.open(mask_path)) > 0
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        if not mask.any():
            raise ValueError(f"Reference mask {mask_path} is empty; INSID3 needs tumour pixels")
        pairs.append((_read_rgb(image_path), mask))
    if not pairs:
        raise FileNotFoundError(
            f"No reference images in {ref_dir}/images. Create them with "
            "make_insid3_references.py (images/ and masks/ with matching file names)."
        )
    return pairs, [os.path.basename(p) for p in image_paths]


@torch.no_grad()
def _encode_references(model, lib, ref_pairs, device):
    """Everything predict_mask() computes from the references alone, done once."""
    transform = model._transform
    ref_images = torch.cat([transform(img).unsqueeze(0) for img, _ in ref_pairs]).to(device)
    ref_masks = torch.stack([
        F.interpolate(torch.from_numpy(m)[None, None].float(), size=(IMAGE_SIZE, IMAGE_SIZE),
                      mode="nearest")[0, 0] > 0.5
        for _, m in ref_pairs
    ]).to(device)                                                  # (S, H, W) bool

    fmaps = model._extract_features(ref_images.unsqueeze(0))      # (1, S, C, h, w)
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    feat_refs_deb = model._debias_features(fmaps_norm)            # (1, S, C, h, w)
    h, w = feat_refs_deb.shape[-2:]
    ref_masks = ref_masks.unsqueeze(1)                             # (S, 1, H, W)

    prototypes = []
    for s in range(ref_masks.shape[0]):
        mask_s = lib["downsample_mask"](ref_masks[s:s + 1], h, w)
        fg = feat_refs_deb[0, s, :, mask_s]
        if fg.shape[1] > 0:
            prototypes.append(fg.mean(dim=1))
    if not prototypes:
        raise ValueError("No reference mask covers a single feature patch; use larger tumours.")
    ref_prototype = F.normalize(torch.stack(prototypes).mean(dim=0), p=2, dim=0).unsqueeze(1)
    return {"feat_refs_deb": feat_refs_deb, "ref_masks": ref_masks, "ref_prototype": ref_prototype}


@torch.no_grad()
def _segment(model, lib, refs, tgt_image, orig_hw):
    """INSID3.predict_mask() with the reference features precomputed.

    Mirrors models/insid3.py (fork-changes, de390a0) step for step; only the
    reference half of the feature extraction is cached. Returns a bool (H, W) mask
    at the original frame size.
    """
    feat_refs_deb = refs["feat_refs_deb"]
    ref_masks = refs["ref_masks"]
    ref_prototype = refs["ref_prototype"]
    S = feat_refs_deb.shape[1]

    fmaps = model._extract_features(tgt_image.unsqueeze(0))       # (1, 1, C, h, w)
    fmaps_norm = F.normalize(fmaps, p=2, dim=2)
    _, _, C, h, w = fmaps_norm.shape
    feat_tgt = fmaps_norm[:, 0]
    feat_tgt_deb = model._debias_features(fmaps_norm)[:, 0]

    sim_maps = [torch.einsum("bchw,bcxy->bhwxy", feat_refs_deb[:, m], feat_tgt_deb) for m in range(S)]
    candidate_mask = model._locate_candidates(sim_maps, ref_masks, feat_tgt_deb, ref_prototype, h, w)

    if candidate_mask.sum() > 0:
        feat_tgt_flat = feat_tgt[0].reshape(C, -1).permute(1, 0)
        cluster_labels = lib["agglomerative_clustering"](feat_tgt_flat, model.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1
        feat_tgt_deb_flat = feat_tgt_deb[0].reshape(C, -1).permute(1, 0)
        cluster_protos = lib["compute_cluster_prototypes"](feat_tgt_deb_flat, cluster_labels.view(-1), K)
        mask = model._seed_and_aggregate(candidate_mask, cluster_labels, cluster_protos, K,
                                         ref_prototype, feat_tgt, feat_tgt_deb, h, w)
    else:
        mask = candidate_mask
    return lib["upsample_mask"](mask, orig_hw[0], orig_hw[1])


def load_model(model_dir, device):
    _install_dependencies()
    lib = _import_insid3()

    backbones = _find_backbones(model_dir)
    if not backbones:
        raise FileNotFoundError(f"No {BACKBONE_GLOB} backbones found under {model_dir}")
    ref_pairs, ref_names = _load_references(model_dir)
    print(f"[insid3_adapter] Folds: {list(backbones)}; {len(ref_pairs)}-shot references: {ref_names}")
    if torch.device(device).type == "cpu":
        print("[insid3_adapter] WARNING: running on CPU; expect many seconds per frame.")

    folds = []
    for fold, (arch, path) in backbones.items():
        encoder = _build_encoder(arch, path)
        model = lib["INSID3"](
            encoder=encoder, image_size=IMAGE_SIZE, svd_components=SVD_COMPONENTS, tau=TAU,
            merge_threshold=MERGE_THRESHOLD, mask_refiner="bilinear",
            resize_to_orig_size=False, device=str(device),
        )
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        refs = _encode_references(model, lib, ref_pairs, device)
        folds.append({"fold": fold, "model": model, "refs": refs})
        print(f"[insid3_adapter] Loaded fold {fold} ({arch}) from {path}")

    return {"folds": folds, "lib": lib, "transform": lib["build_transform"](IMAGE_SIZE)}


def predict(model, input_array, device):
    frame = np.squeeze(np.asarray(input_array)[0])  # current frame, (H, W)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {frame.shape} (input {input_array.shape})")
    orig_hw = frame.shape

    # Works with the module's "Normalize" toggle on or off: it divides by 255, and 0-1
    # input is scaled back to 0-255 here (off avoids a possible 1-grey-level rounding).
    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 else \
        np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(np.stack([frame_uint8] * 3, axis=-1))
    tgt_image = model["transform"](pil_image).unsqueeze(0).to(device)  # (1, 3, 1024, 1024)

    votes = torch.zeros(orig_hw, dtype=torch.float32, device=tgt_image.device)
    for entry in model["folds"]:
        votes += _segment(entry["model"], model["lib"], entry["refs"], tgt_image, orig_hw).float()
    votes /= len(model["folds"])
    return votes.cpu().numpy().astype(np.float32)
