"""
TorchSequenceSegmentation adapter for the OpenUS model -- runs INSIDE SLICER
on native Windows. This is a thin HTTP client, not the model itself: OpenUS
only runs on Linux, so the real model (5-fold ensemble) lives in
openus_adapter.py running inside a WSL2 server (run_wsl2_openus_server.py).
This file just forwards each frame to that server and returns the probability
map it sends back, matching the load_model()/predict() contract
TorchSequenceSegmentation.py's loadCustomModel() expects.

Place this file (ending in "_adapter.py") in the model folder selected in the
Slicer module's model directory setting -- it doesn't need any weights or
checkpoints alongside it, since those live on the WSL2 side.

Before using: start the WSL2 server first (see run_wsl2_openus_server.py's
own docstring for its startup steps) and confirm http://localhost:8000/health
responds from a Windows browser/PowerShell before relying on this in a live
procedure.
"""

import io

import numpy as np
import requests
from PIL import Image

SERVER_URL = "http://localhost:8000"
HEALTH_CHECK_TIMEOUT = 5       # seconds; load_model() fails fast if the server isn't up
PREDICT_TIMEOUT = 10           # seconds; generous for a single frame, but not so long
                                # it freezes the Slicer UI if the server hangs


def load_model(model_dir, device):
    """
    Called once when this model is selected in the Slicer module. Doesn't
    load anything locally -- just confirms the WSL2 server is reachable and
    already has the real model loaded, so failures show up immediately
    rather than silently on the first frame of a live procedure.
    """
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        response.raise_for_status()
        health = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not reach the OpenUS WSL2 server at {SERVER_URL}. "
            f"Make sure run_wsl2_openus_server.py is running inside WSL2 "
            f"(and that WSL2 localhost forwarding is working). Original error: {e}"
        )

    if not health.get("model_loaded", False):
        raise RuntimeError(
            f"OpenUS WSL2 server at {SERVER_URL} is reachable but reports its "
            f"model is not yet loaded (still starting up?). Wait for the "
            f"server's own log to print \"Model loaded.\" and try again."
        )

    print(f"[openus_wsl2_client_adapter] Connected to OpenUS server at {SERVER_URL}")
    return {"server_url": SERVER_URL}


def predict(model, input_array, device):
    """
    Sends the current frame to the WSL2 server and returns the ensembled
    foreground probability map it computes, as a float32 (H, W) array with
    values in [0, 1] -- matching the rest of this module's adapter contract.
    """
    server_url = model["server_url"]

    frame = np.squeeze(input_array[0])  # current frame; squeeze out any singleton axes
    if frame.ndim != 2:
        raise ValueError(
            f"Expected a single 2D (H, W) ultrasound frame, but got shape {frame.shape} "
            f"after squeezing input_array[0] (shape {input_array[0].shape}; full "
            f"input_array shape: {input_array.shape}). This usually means the selected "
            f"input volume/sequence hasn't started streaming real frames yet (e.g. "
            f"Slicer's placeholder volume before an IGTL connection/sequence actually "
            f"has data) -- check the input volume's real dimensions before predict() "
            f"is called, rather than only here."
        )

    # Match the assumption on the server side (openus_adapter.predict):
    # treat values >1.5 as already 0-255, otherwise assume 0-1 and rescale.
    # As noted when this adapter was set up, keep the module's own
    # "Normalize" toggle OFF so this frame arrives as raw intensities.
    frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1.5 else \
        np.clip(frame * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(frame_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    try:
        files = {"file": ("frame.png", buf, "image/png")}
        response = requests.post(f"{server_url}/predict", files=files, timeout=PREDICT_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"OpenUS WSL2 server request failed: {e}. "
            f"Check that run_wsl2_openus_server.py is still running inside WSL2."
        )

    probability_png = Image.open(io.BytesIO(response.content))
    output_array = np.array(probability_png).astype(np.float32) / 255.0
    return output_array