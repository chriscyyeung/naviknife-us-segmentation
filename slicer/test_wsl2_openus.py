"""
Runs on native Windows (e.g. from Slicer's Python console, or your
real-time application code). Sends an ultrasound frame to the OpenUS
inference server running inside WSL2, and gets a foreground probability
map back (float32, values in [0, 1]).

Requires `requests` in Slicer's Python environment:
    pip install requests
(Slicer's Python console can run pip installs the same way as any
other Python: use slicer's pip_install() helper, or a terminal pointed
at Slicer's own Python executable.)
"""

import io

import numpy as np
import requests
from PIL import Image

SERVER_URL = "http://localhost:8000"


def check_server_health():
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def predict(image_arr: np.ndarray) -> np.ndarray:
    """Send a single image (H, W, 3) or (H, W) array to the WSL2 server,
    return the foreground probability map as a float32 array (H, W) with
    values in [0, 1]."""
    img = Image.fromarray(image_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("frame.png", buf, "image/png")}
    response = requests.post(f"{SERVER_URL}/predict", files=files, timeout=30)
    response.raise_for_status()

    probability_png = Image.open(io.BytesIO(response.content))
    return np.array(probability_png).astype(np.float32) / 255.0


if __name__ == "__main__":
    print("Server health:", check_server_health())

    # Quick end-to-end test with one of your existing frames
    test_image_path = "LumpNavPNGResMatched/images/LN003_frame00000.png"
    arr = np.array(Image.open(test_image_path).convert("RGB"))

    mask = predict(arr)
    print(f"Received probability map: shape={mask.shape}, "
          f"min={mask.min():.3f}, max={mask.max():.3f}, mean={mask.mean():.3f}")

    preview = (mask * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(preview, mode="L").save("LN003_frame00000_wsl2_pred.png")
    print("Saved LN003_frame00000_wsl2_pred.png")