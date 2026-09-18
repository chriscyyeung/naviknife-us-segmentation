"""
Template adapter script for TorchSequenceSegmentation.

Copy this file into the same folder as your model's weights, rename it to
something ending in "_adapter.py" (e.g. "my_model_adapter.py"), and edit
load_model() / predict() below to match your model. The module will
discover it automatically alongside any TorchScript ".pt" models in the
folder you select as the model directory.

Only two functions are required. Everything about how the model is built
and how its output is turned into a single-channel prediction is up to you,
so this works for any architecture, any framework, and any custom pre/post
processing.
"""

import os
import re
import torch
import numpy as np


def load_model(model_dir, device):
    """
    Called once when the model is selected.

    model_dir: absolute path to the folder containing this adapter script
               and your weight file(s).
    device:    torch.device to use ("cuda:0" or "cpu").

    Build/load your model here however you need to (torch.load with a
    state_dict, an ONNX Runtime session, a scikit-learn pickle, etc.) and
    return whatever object predict() below expects as its "model" argument.
    """
    # install nnUNet
    try:
        import nnunetv2
    except ImportError:
        import slicer
        slicer.util.pip_install("nnunetv2")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = os.path.join(model_dir, "nnUNetTrainer__nnUNetPlans__2d")

    f = []
    for entry in os.listdir(model_folder):
        full_path = os.path.join(model_folder, entry)
        if not os.path.isdir(full_path):
            continue
        match = re.fullmatch(r"fold_(\d+)", entry)
        if match:
            f.append(int(match.group(1)))

    predictor = nnUNetPredictor(tile_step_size=0.5,
                                use_gaussian=True,
                                use_mirroring=False,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=False,
                                verbose_preprocessing=True,
                                allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        f,
        checkpoint_name="checkpoint_final.pth"
    )
    return predictor


def predict(model, input_array, device):
    """
    Called once per frame.

    model:       the object returned by load_model().
    input_array: float32 numpy array, shape (C, H, W), where C = 1 + the
                 number of previous frames requested in the GUI (usually
                 C = 1 unless you're using the "previous frames" buffer).
    device:      torch.device to use.

    Must return a single-channel float numpy array of shape (H, W)
    representing the foreground prediction (e.g. a probability map, values
    roughly in [0, 1]). The rest of the pipeline (flipping, log transform,
    scan conversion, thresholding) is handled outside this function, so you
    only need to worry about getting from raw input pixels to a prediction
    map here.
    """
    image_properties = {"spacing": [999.0, 1.0, 1.0]}
    seg, probabilities = model.predict_single_npy_array(
        input_array, 
        image_properties=image_properties,
        save_or_return_probabilities=True
    )
    output_array = probabilities[1, 0, :, :]
    return output_array
