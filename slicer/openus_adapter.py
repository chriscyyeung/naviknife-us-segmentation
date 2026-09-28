"""
Adapter for an OpenUS (vmamba_small backbone + MambaDecoder head) model,
fine-tuned with 5-fold cross-validation, ensembled at inference time.
"""

import os
import re
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---- Configuration -- adjust these for your environment/checkpoint layout ----
OPENUS_REPO_DIR = "c:/Users/chris/Repos/naviknife-us-segmentation/external/OpenUS"
PRETRAINED_VMAMBA = "vmamba/vssm_small_0229_ckpt_epoch_222.pth"
BACKBONE_CHECKPOINT_NAME = "openus_cpt0150.pth"
BACKBONE_CHECKPOINT_KEY = "teacher"   # key inside the backbone checkpoint dict
HEAD_CHECKPOINT_NAME = "checkpoint_teacher_seg_best.pth"
NUM_CLASSES = 2
FOREGROUND_CLASS_INDEX = 1
INPUT_SIZE = 224  # exposed for the module's optional metadata convention


def _get_fold_head_checkpoints(model_dir):
    """Mirrors the nnU-Net fold-discovery helper: find fold_N subfolders,
    each expected to contain a head checkpoint."""
    fold_checkpoints = {}
    for entry in os.listdir(model_dir):
        full_path = os.path.join(model_dir, entry)
        match = re.fullmatch(r"fold_(\d+)", entry)
        if not match or not os.path.isdir(full_path):
            continue
        ckpt_path = os.path.join(full_path, HEAD_CHECKPOINT_NAME)
        if os.path.isfile(ckpt_path):
            fold_checkpoints[int(match.group(1))] = ckpt_path
    return dict(sorted(fold_checkpoints.items()))


def load_model(model_dir, device):
    # Make the OpenUS repo's local packages (models, vmamba_models, dataset, utils)
    # importable -- this is a plain research repo, not pip-installable.
    if OPENUS_REPO_DIR not in sys.path:
        sys.path.insert(0, OPENUS_REPO_DIR)

    # install dependencies (torch, torchvision, timm, etc.) if not already present
    try:
        import fvcore
        import triton
    except ImportError:
        import slicer
        slicer.util.pip_install("fvcore, triton-windows")

    from vmamba_models.dino_vmamba import Backbone_DINOv2_VSSM_2
    from dataset.transforms import get_transforms

    # ---- Shared, frozen backbone (built once, reused by every fold's head) ----
    vmamba_ckpt_path = os.path.join(model_dir, PRETRAINED_VMAMBA)
    backbone = Backbone_DINOv2_VSSM_2(pretrained=vmamba_ckpt_path, seg_head=True)  # weights loaded manually below
    backbone_ckpt_path = os.path.join(model_dir, BACKBONE_CHECKPOINT_NAME)
    backbone_checkpoint = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = backbone_checkpoint[BACKBONE_CHECKPOINT_KEY]
    clean_state_dict = {
        k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")
    }
    msg = backbone.load_state_dict(clean_state_dict, strict=False)
    print(f"[openus_adapter] Loaded shared backbone, load_state_dict msg: {msg}")
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # ---- One MambaDecoderHead per fold ----
    from vmamba_models.MambaDecoder import MambaDecoder
    from torch import nn

    class MambaDecoderHead(nn.Module):
        # Duplicated here (rather than imported) since it's defined inline in
        # test_segmentation.py, not exported from the repo's own modules.
        def __init__(self, num_classes):
            super().__init__()
            self.decoder = MambaDecoder(num_classes=num_classes)

        def forward(self, encoder_out, orig_hw):
            logits = self.decoder.forward(encoder_out)
            return F.interpolate(logits, size=orig_hw, mode="bilinear", align_corners=False)

    fold_checkpoints = _get_fold_head_checkpoints(model_dir)
    if not fold_checkpoints:
        raise FileNotFoundError(
            f"No fold_N/{HEAD_CHECKPOINT_NAME} checkpoints found under {model_dir}"
        )
    print(f"[openus_adapter] Found folds: {list(fold_checkpoints.keys())}")

    heads = []
    for fold_num, ckpt_path in fold_checkpoints.items():
        head = MambaDecoderHead(num_classes=NUM_CLASSES)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        head.load_state_dict(checkpoint["state_dict"])
        head.to(device)
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        heads.append(head)
        print(f"[openus_adapter] Loaded fold {fold_num} head from {ckpt_path} "
              f"(epoch {checkpoint.get('epoch', 'N/A')}, "
              f"best_dice {checkpoint.get('best_dice', 'N/A')})")

    # --- Alternative, if your folds fine-tuned the backbone too ("complete" checkpoints
    # with 'model_state_dict' + 'head_state_dict' or 'complete_model_state_dict'):
    # build and load a SEPARATE backbone per fold instead of sharing one, and pair each
    # with its own head in the returned list. Ask if you need this variant fleshed out.

    _, test_transform = get_transforms(img_size=INPUT_SIZE)

    return {"backbone": backbone, "heads": heads, "transform": test_transform}


def predict(model, input_array, device):
    backbone = model["backbone"]
    heads = model["heads"]
    transform = model["transform"]

    frame = input_array[0]  # current frame, (H, W)
    orig_h, orig_w = frame.shape

    # Grayscale -> 3-channel uint8 PIL image, since the backbone is ImageNet/DINO-style
    # pretrained and get_transforms' pipeline expects a PIL Image as input.
    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 else \
        np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    rgb_frame = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1)  # (H, W, 3)
    pil_image = Image.fromarray(rgb_frame)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.inference_mode():
        encoder_out = backbone(input_tensor)

        fold_probs = []
        for head in heads:
            logits = head(encoder_out, orig_hw=input_tensor.shape[2:])          # (1, num_classes, 224, 224)
            probs = torch.softmax(logits, dim=1)[0, FOREGROUND_CLASS_INDEX]     # (224, 224)
            fold_probs.append(probs)

        ensembled_probs = torch.stack(fold_probs, dim=0).mean(dim=0)  # soft-vote average across folds

        # Resize the 224x224 prediction back up to the original frame resolution
        ensembled_probs = ensembled_probs.unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)
        ensembled_probs = F.interpolate(
            ensembled_probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        output_array = ensembled_probs.squeeze(0).squeeze(0).cpu().numpy()

    return output_array.astype(np.float32)
