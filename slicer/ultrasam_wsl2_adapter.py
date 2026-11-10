"""
TorchSequenceSegmentation adapter for UltraSAM that runs INSIDE SLICER on Windows.
It is a thin HTTP client: the real model (ultrasam_adapter.py, 5-fold ensemble)
runs in WSL2 behind run_wsl2_adapter_server.py, because mmcv/mmdet do not install
cleanly into Slicer's Python.

Put this file alone in a model folder (e.g. models/UltraSAM_WSL2/) and select that
folder in the module. Start the server first:

    python run_wsl2_adapter_server.py --adapter ultrasam_adapter.py \
        --model-dir /path/to/ultrasam_work_dirs --port 8001

The module's "Normalize" toggle can be on or off: it divides by 255, and 0-1 frames
are scaled back to 0-255 before sending (off avoids a 1-grey-level rounding).
"""

import io
import os

import numpy as np
import requests
from PIL import Image

SERVER_URL = os.environ.get("ULTRASAM_SERVER_URL", "http://127.0.0.1:8001")
MODEL_NAME = "UltraSAM"
HEALTH_CHECK_TIMEOUT = 5   # seconds; load_model() fails fast if the server isn't up
PREDICT_TIMEOUT = 30       # seconds; 5 SAM ViT-B folds per frame


def load_model(model_dir, device):
    # trust_env=False skips Windows proxy auto-detection, which can add seconds to
    # every localhost request (the slow-request issue seen with the OpenUS server).
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"{SERVER_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        response.raise_for_status()
        health = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the {MODEL_NAME} server at {SERVER_URL}. Start "
            f"run_wsl2_adapter_server.py in WSL2 first. Original error: {e}"
        )
    if not health.get("model_loaded", False):
        raise RuntimeError(f"{MODEL_NAME} server at {SERVER_URL} is up but its model is not loaded yet; "
                           "wait for \"Model loaded.\" in the server log.")
    print(f"[ultrasam_wsl2_adapter] Connected to {health.get('adapter')} at {SERVER_URL}")
    return {"session": session}


def predict(model, input_array, device):
    frame = np.squeeze(np.asarray(input_array)[0])
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2D frame, got shape {frame.shape} (input {input_array.shape}). "
                         "The input volume may not be streaming real frames yet.")
    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 else \
        np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(frame_uint8).save(buf, format="PNG")
    buf.seek(0)
    try:
        response = model["session"].post(f"{SERVER_URL}/predict",
                                          files={"file": ("frame.png", buf, "image/png")},
                                          timeout=PREDICT_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"{MODEL_NAME} server request failed: {e}. Is the WSL2 server still running?")
    return np.array(Image.open(io.BytesIO(response.content))).astype(np.float32) / 255.0
