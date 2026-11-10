"""
Serves any TorchSequenceSegmentation adapter (load_model/predict) over HTTP, so a
model whose dependencies only install on Linux can run inside WSL2 while Slicer
runs on Windows. Generalises run_wsl2_openus_server.py (same /predict and /health
API), so one thin client adapter works for every model.

Usage, inside WSL2 in the model's own environment:

    pip install fastapi uvicorn python-multipart pillow numpy
    python run_wsl2_adapter_server.py --adapter ultrasam_adapter.py \
        --model-dir /path/to/ultrasam_work_dirs --port 8001

Then put the matching client (e.g. ultrasam_wsl2_adapter.py, pointing at the same
port) in the Slicer model folder. Check http://127.0.0.1:<port>/health from
Windows before running a procedure.

Suggested ports: 8000 OpenUS (existing server), 8001 UltraSAM, 8002 INSID3.
"""

import argparse
import importlib.util
import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI()
state = {"adapter": None, "model": None, "device": None, "name": None}


def _load_adapter(path):
    path = os.path.abspath(path)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return name, module


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Takes a PNG frame, returns the foreground probability map as an 8-bit PNG
    (pixel / 255 = probability), same format as run_wsl2_openus_server.py."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is still loading, try again shortly.")
    img = Image.open(io.BytesIO(await file.read()))
    arr = np.array(img.convert("L") if img.mode not in ("L", "I;16", "I", "F") else img).astype(np.float32)
    probability = state["adapter"].predict(state["model"], arr[None], state["device"])
    out = np.clip(np.asarray(probability, dtype=np.float32) * 255.0, 0, 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out, mode="L").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None, "adapter": state["name"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="path to a *_adapter.py with load_model/predict")
    p.add_argument("--model-dir", required=True, help="passed to the adapter's load_model()")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup-shape", type=int, nargs=2, default=[525, 615], metavar=("H", "W"))
    args = p.parse_args()

    state["name"], state["adapter"] = _load_adapter(args.adapter)
    state["device"] = torch.device(args.device)
    print(f"Loading {state['name']} from {args.model_dir} on {state['device']}...")
    model = state["adapter"].load_model(args.model_dir, state["device"])
    print("Warming up...")
    state["adapter"].predict(model, np.zeros([1] + args.warmup_shape, dtype=np.float32), state["device"])
    state["model"] = model
    print("Model loaded.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
