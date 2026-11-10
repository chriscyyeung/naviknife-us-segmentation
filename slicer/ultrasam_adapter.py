"""
TorchSequenceSegmentation adapter for UltraSAM (SAM ViT-B, mmdet) fine-tuned on
LumpNav with a constant full-image box prompt, ensembled over the 5 CV folds.

No user prompt is needed: the inference config's GetFullSizeBox transform
prompts every frame with a box covering the whole image, as in training and in
the cross-validation evaluation.

UltraSAM needs mmcv/mmdet/mmpretrain (and transformers<4.56), which are hard to
install into Slicer's Python on Windows. The intended setup is the same as
OpenUS: run this file inside WSL2 behind run_wsl2_adapter_server.py, and put
ultrasam_wsl2_adapter.py in the Slicer model folder. It also works directly in
Slicer if those packages are installed there.

Model folder layout (mmengine work_dirs copied from the cluster):

    <model_dir>/
        UltraSAM_box_refine_fold0/iter_NNNNN.pth   (or best_*.pth)
        ...
        UltraSAM_box_refine_fold4/iter_NNNNN.pth

Any folder whose name ends in fold_<k> or fold<k> counts as fold k.

Requires the UltraSam fork at branch fork-changes (commit 4d47a99 "fixed static
bbox prompt for inference" or later). Older checkouts, such as 88ff0a2, make a
box of a couple of pixels instead of the full image; load_model() checks the
prompt and refuses to run if it does not cover the frame.
"""

import glob
import os
import re
import sys

import numpy as np
import torch

# ---- Configuration -- adjust these for your environment/checkpoint layout ----
ULTRASAM_REPO_DIR = os.environ.get(
    "ULTRASAM_REPO_DIR",
    "/home/chris/projects/naviknife-us-segmentation/external/UltraSam",
)
# Relative to ULTRASAM_REPO_DIR unless absolute. Prompt-only test pipeline (no GT needed).
CONFIG_PATH = os.environ.get(
    "ULTRASAM_CONFIG", "configs/UltraSAM/UltraSAM_full/UltraSAM_box_inference.py"
)
CHECKPOINT_NAME = os.environ.get("ULTRASAM_CHECKPOINT", "")  # e.g. "iter_30000.pth"; empty = auto
FOLDS = None            # None = every fold found; e.g. [0] for one fold
SCORE_THRESHOLD = 0.0   # instances kept and unioned, as in pkl_preds_to_masks.py
WARMUP_SHAPE = (525, 615)  # (H, W) of a LumpNav frame, used for the prompt check

_FOLD_DIR_RE = re.compile(r"fold_?(\d+)$")
_ITER_RE = re.compile(r"iter_(\d+)\.pth$")
# Prompt transforms that read the ground truth mask/box; they cannot run on live frames
_GT_TRANSFORMS = {"LoadAnnotations", "GetPointFromMask", "GetPointBox", "GetPointFromBox"}


def _pick_checkpoint(fold_dir):
    if CHECKPOINT_NAME:
        path = os.path.join(fold_dir, CHECKPOINT_NAME)
        return path if os.path.isfile(path) else None
    best = sorted(glob.glob(os.path.join(fold_dir, "best_*.pth")))
    if best:
        return best[-1]
    iters = [(int(_ITER_RE.search(p).group(1)), p)
             for p in glob.glob(os.path.join(fold_dir, "iter_*.pth")) if _ITER_RE.search(p)]
    return max(iters)[1] if iters else None


def _find_checkpoints(model_dir):
    found = {}
    for d in [model_dir] + glob.glob(os.path.join(model_dir, "**", "*"), recursive=True):
        if not os.path.isdir(d):
            continue
        match = _FOLD_DIR_RE.search(os.path.basename(os.path.normpath(d)))
        if not match:
            continue
        ckpt = _pick_checkpoint(d)
        if ckpt is None:
            continue
        fold = int(match.group(1))
        if fold in found:
            raise ValueError(f"Two checkpoints for fold {fold}: {found[fold]} and {ckpt}. "
                             "Remove one so folds are unambiguous.")
        found[fold] = ckpt
    if FOLDS is not None:
        found = {k: v for k, v in found.items() if k in FOLDS}
    return dict(sorted(found.items()))


def _base_type(t):
    return str(t).split(".")[-1]


def _build_test_pipeline(cfg):
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    pipeline_cfg = get_test_pipeline_cfg(cfg.copy())
    kept = []
    for step in pipeline_cfg:
        name = _base_type(step["type"])
        if name == "LoadAnnotations":
            continue  # harmless to drop: nothing downstream may read the GT
        if name in _GT_TRANSFORMS:
            raise ValueError(f"The UltraSAM config's test pipeline uses {name}, which prompts from the "
                             f"ground truth. Use the full-image box config ({CONFIG_PATH}).")
        kept.append(step)
    if _base_type(kept[0]["type"]) != "LoadImageFromFile":
        raise ValueError(f"Unexpected first test transform: {kept[0]['type']}")
    kept[0] = dict(type="mmdet.LoadImageFromNDArray")
    if not any(_base_type(s["type"]) == "GetFullSizeBox" for s in kept):
        raise ValueError("The test pipeline has no GetFullSizeBox prompt; this adapter needs one.")
    return Compose(kept)


def _check_prompt(result, frame_hw):
    """Same check as scripts/UltraSAM/test_inference.py: the box prompt, mapped back
    to original pixels, should cover the whole frame."""
    gt = result.gt_instances
    if "boxes" not in gt:
        raise RuntimeError("UltraSAM result carries no box prompt; check the test pipeline.")
    sx, sy = result.metainfo["scale_factor"]
    box = gt.boxes.detach().cpu().numpy().reshape(-1, 2, 2)[0].astype(np.float64)
    x1, x2 = (box[:, 0] - 0.5) / sx
    y1, y2 = (box[:, 1] - 0.5) / sy
    h, w = frame_hw
    coverage = max(x2 - x1, 0) * max(y2 - y1, 0) / (h * w)
    print(f"[ultrasam_adapter] Box prompt ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f}) "
          f"covers {coverage:.0%} of a {w}x{h} frame")
    if not 0.9 <= coverage <= 1.1:
        raise RuntimeError(
            "UltraSAM's full-image box prompt does not cover the frame. The UltraSam checkout is "
            "probably older than the GetFullSizeBox fix: check out branch fork-changes (4d47a99 or later)."
        )


def load_model(model_dir, device):
    if ULTRASAM_REPO_DIR not in sys.path:
        sys.path.insert(0, ULTRASAM_REPO_DIR)
    try:
        import mmdet  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "UltraSAM needs mmcv, mmdet and mmpretrain (transformers<4.56). Run it in the UltraSAM "
            "environment behind run_wsl2_adapter_server.py, and use ultrasam_wsl2_adapter.py in Slicer."
        ) from e

    import torch.nn.functional as F
    from mmengine.config import Config
    from mmdet.apis import inference_detector, init_detector
    # Same monkey patch as UltraSAM's own inference scripts (test_inference.py)
    from endosam.models.utils.custom_functional import multi_head_attention_forward
    F.multi_head_attention_forward = multi_head_attention_forward

    config_path = CONFIG_PATH if os.path.isabs(CONFIG_PATH) else os.path.join(ULTRASAM_REPO_DIR, CONFIG_PATH)
    checkpoints = _find_checkpoints(model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No fold_<k> folders with UltraSAM .pth checkpoints under {model_dir}")
    print(f"[ultrasam_adapter] Config {config_path}; folds {list(checkpoints)}")

    # Relative paths inside the configs (custom_imports, weights/) resolve from the repo
    cwd = os.getcwd()
    os.chdir(ULTRASAM_REPO_DIR)
    try:
        cfg = Config.fromfile(config_path)
        models = []
        for fold, ckpt in checkpoints.items():
            model = init_detector(cfg, ckpt, device=str(device))
            model.eval()
            models.append(model)
            print(f"[ultrasam_adapter] Loaded fold {fold} from {ckpt}")
        pipeline = _build_test_pipeline(models[0].cfg)
    finally:
        os.chdir(cwd)

    dummy = np.zeros(WARMUP_SHAPE + (3,), dtype=np.uint8)
    _check_prompt(inference_detector(models[0], dummy, test_pipeline=pipeline), WARMUP_SHAPE)

    return {"models": models, "pipeline": pipeline, "inference_detector": inference_detector}


def predict(model, input_array, device):
    frame = np.squeeze(np.asarray(input_array)[0])  # current frame, (H, W)
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {frame.shape} (input {input_array.shape})")
    h, w = frame.shape

    # Raw 0-255 intensities expected (module's "Normalize" toggle OFF); 0-1 input is rescaled.
    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 else \
        np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(np.stack([frame_uint8] * 3, axis=-1))  # grey, so BGR == RGB

    votes = np.zeros((h, w), dtype=np.float32)
    with torch.inference_mode():
        for detector in model["models"]:
            result = model["inference_detector"](detector, image, test_pipeline=model["pipeline"])
            inst = result.pred_instances
            mask = np.zeros((h, w), dtype=bool)
            if "masks" in inst and len(inst) > 0:
                keep = (inst.scores >= SCORE_THRESHOLD).cpu().numpy()
                masks = inst.masks.cpu().numpy() if hasattr(inst.masks, "cpu") else np.asarray(inst.masks)
                if keep.any():
                    mask = masks[keep].reshape(-1, h, w).any(axis=0)
            votes += mask
    return votes / len(model["models"])
