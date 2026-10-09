"""
Runs inside WSL2. Loads the OpenUS/vmamba model once at startup and
exposes a single /predict endpoint that Slicer (on native Windows) can
call over localhost to get a segmentation mask back.

Usage (inside WSL2, in the environment with triton/vmamba/OpenUS installed):
    pip install fastapi uvicorn python-multipart pillow numpy
    # openus_ensemble_adapter.py must be in the same directory as this file
    export OPENUS_MODEL_DIR=/path/to/your/openus_model_dir  # backbone + fold_0..fold_4
    python run_wsl2_openus_server.py

This starts listening on http://localhost:8000 . Recent WSL2 versions
forward localhost automatically between Windows and WSL2 (check with
`wsl --status`; if it's not working, mirrored networking mode -- set
in %UserProfile%\\.wslconfig -- makes this reliable).
"""

import io
import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# openus_adapter.py (the TorchSequenceSegmentation adapter script) is
# reused directly here, so the WSL2 server and the Slicer module load/predict
# using the exact same code path rather than two independently-maintained
# copies of the model logic.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openus_adapter import load_model as load_openus_model
from openus_adapter import predict as predict_openus

app = FastAPI()

# ---- Configuration ----------------------------------------------------------
# Path (inside WSL2) to the OpenUS model directory: the shared backbone
# checkpoint plus fold_0 .. fold_4 head checkpoints, as expected by
# openus_adapter.load_model(). Override with an env var so this
# doesn't need to be hardcoded per machine.
MODEL_DIR = os.environ.get("OPENUS_MODEL_DIR", "/path/to/your/openus_model_dir")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---- Load model once at startup, not per-request --------------------------
model = None


@app.on_event("startup")
def load_model_once():
    global model
    print(f"Loading OpenUS model from {MODEL_DIR} on device {DEVICE}...")
    model = load_openus_model(MODEL_DIR, DEVICE)
    print("Model loaded.")

    print("Warming up (first run compiles Triton kernels)...")
    dummy = np.zeros((1, 615, 525), dtype=np.float32)  # use your real frame size
    predict_openus(model, dummy, DEVICE)
    print("Warmup done.")


def run_inference(image_arr: np.ndarray) -> np.ndarray:
    """Run the model on a single image, return the ensembled foreground
    probability map as a float32 array (H, W) with values in [0, 1]."""
    if model is None:
        raise RuntimeError("Model has not finished loading yet.")

    # image_arr arrives as (H, W, 3) RGB (see the /predict handler below);
    # the source is a grayscale ultrasound frame duplicated across channels,
    # so collapsing back to single-channel with a mean is lossless here.
    grayscale = image_arr.mean(axis=-1) if image_arr.ndim == 3 else image_arr

    # openus_adapter.predict() expects (C, H, W) with C=1 (current
    # frame only, no previous-frame buffer in this server context).
    input_array = grayscale[None, :, :].astype(np.float32)

    probability_map = predict_openus(model, input_array, DEVICE)  # (H, W) float32, foreground prob
    return probability_map


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts a PNG image, returns the foreground probability map as an
    8-bit grayscale PNG (mode 'L'): pixel value / 255.0 recovers the
    probability. Quantizing to 8 bits matches the precision already used
    elsewhere in this pipeline (see the *255 scaling in getPrediction());
    switch to 16-bit ('I;16') here and on the client if finer precision
    ever turns out to matter."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    arr = np.array(img)

    probability_map = run_inference(arr)
    probability_png = np.clip(probability_map * 255.0, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(probability_png, mode="L").save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)