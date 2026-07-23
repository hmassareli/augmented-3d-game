"""
Depth Anything 3 Metric backend.

Receives a video frame (JPEG/PNG) via POST /api/depth,
runs DA3-Metric-Large inference, returns:
  - depth map (relative, normalized 0-1) as PNG
  - metric depth values at requested pixel coordinates (JSON)

The web client sends wrist pixel coordinates from MediaPipe
and gets back the metric depth (in meters) at those pixels.
"""

from __future__ import annotations

import io
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

app = FastAPI(title="DA3 Depth Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):517[3-9]",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Config
MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
INFERENCE_SIZE = (518, 518)  # (H, W) — keeps speed high, quality sufficient
FOCAL_FALLBACK = 800.0  # px, typical webcam focal length if not provided

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_is_loading = False


def get_model():
    """Lazy-load DA3 model on first request."""
    global _model, _is_loading
    if _model is not None:
        return _model
    if _is_loading:
        return None
    _is_loading = True
    try:
        from depth_anything_3.api import DepthAnything3

        print(f"[DA3] Loading {MODEL_NAME} on {_device}...")
        _model = DepthAnything3.from_pretrained(MODEL_NAME)
        _model = _model.to(device=_device, dtype=torch.float32)
        _model.eval()
        print(f"[DA3] Model loaded. Inference size: {INFERENCE_SIZE}")
    except Exception as e:
        print(f"[DA3] Failed to load model: {e}")
        print("[DA3] Install with: pip install -e . && pip install xformers")
        _model = None
    finally:
        _is_loading = False
    return _model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_depth_inference(
    image_bgr: np.ndarray,
    focal_px: float = FOCAL_FALLBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run DA3 metric depth inference on a single BGR image.

    Returns:
        metric_depth: (H, W) float32 array in meters
        relative_depth: (H, W) float32 array normalized 0-1 (for visualization)
    """
    model = get_model()
    if model is None:
        raise RuntimeError("DA3 model not loaded")

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Resize to inference size
    pil_resized = pil_image.resize(
        (INFERENCE_SIZE[1], INFERENCE_SIZE[0]), Image.LANCZOS
    )
    img_np = np.array(pil_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(_device)

    with torch.no_grad():
        prediction = model.inference([pil_resized])

    # DA3 metric output: metric_depth = focal * net_output / 300.
    raw_depth = prediction.depth[0]  # (H, W) float32
    metric_depth = focal_px * raw_depth / 300.0

    # Relative depth for visualization (0-1)
    rel_min, rel_max = float(raw_depth.min()), float(raw_depth.max())
    if rel_max - rel_min > 1e-6:
        relative_depth = (raw_depth - rel_min) / (rel_max - rel_min)
    else:
        relative_depth = np.zeros_like(raw_depth)

    return metric_depth, relative_depth


def sample_depth_at_points(
    depth_map: np.ndarray,
    points: list[dict[str, float]],
    original_size: tuple[int, int],
) -> list[float]:
    """
    Sample depth values at (x, y) pixel coordinates in the original image space.

    Args:
        depth_map: (H, W) depth array at inference resolution
        points: list of {"x": 0..1, "y": 0..1} normalized coordinates
        original_size: (height, width) of the original image

    Returns:
        list of depth values (meters) at each point
    """
    dh, dw = depth_map.shape
    results = []
    for pt in points:
        # Points are normalized 0-1 relative to original image
        px = int(pt["x"] * dw)
        py = int(pt["y"] * dh)
        px = max(0, min(dw - 1, px))
        py = max(0, min(dh - 1, py))

        # Sample a small 5x5 window and take median for robustness
        x0 = max(0, px - 2)
        x1 = min(dw, px + 3)
        y0 = max(0, py - 2)
        y1 = min(dh, py + 3)
        window = depth_map[y0:y1, x0:x1]
        results.append(float(np.median(window)))

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    model = get_model()
    return {
        "ready": model is not None,
        "model": MODEL_NAME if model is not None else "loading",
        "device": str(_device),
        "inference_size": list(INFERENCE_SIZE),
    }


@app.post("/api/depth")
async def depth(
    image: UploadFile = File(...),
    points: str = Form(default="[]"),
    focal: str = Form(default=str(FOCAL_FALLBACK)),
) -> JSONResponse:
    """
    Run depth estimation on an uploaded image.

    Args:
        image: JPEG/PNG frame from webcam
        points: JSON array of {"x": 0..1, "y": 0..1} normalized coordinates
                (e.g. wrist positions from MediaPipe)
        focal: Focal length in pixels (for metric depth conversion)

    Returns:
        JSON with:
          - depths: list of metric depth values (meters) at each point
          - depth_image: base64-encoded depth visualization PNG
          - inference_ms: inference time in milliseconds
    """
    import json
    import base64

    # Read image
    img_bytes = await image.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Could not decode image"},
        )

    h, w = img_bgr.shape[:2]
    pts = json.loads(points) if points else []
    focal_px = float(focal) if focal else FOCAL_FALLBACK

    # Run inference
    t0 = time.perf_counter()
    try:
        metric_depth, rel_depth = run_depth_inference(img_bgr, focal_px)
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)},
        )
    inference_ms = (time.perf_counter() - t0) * 1000

    # Sample depth at requested points
    depths = sample_depth_at_points(
        metric_depth, pts, (h, w)
    ) if pts else []

    # Encode depth visualization as JPEG
    depth_vis = (rel_depth * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    _, jpeg_buf = cv2.imencode(".jpg", depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 70])
    depth_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    return JSONResponse(content={
        "depths": depths,
        "depth_image": f"data:image/jpeg;base64,{depth_b64}",
        "inference_ms": round(inference_ms, 1),
        "depth_shape": list(metric_depth.shape),
    })


@app.post("/api/depth/raw")
async def depth_raw(
    image: UploadFile = File(...),
    focal: str = Form(default=str(FOCAL_FALLBACK)),
) -> Response:
    """
    Run depth estimation and return the depth map as a 16-bit PNG.
    Useful for debugging and visualization.
    """
    img_bytes = await image.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Could not decode image"},
        )

    focal_px = float(focal) if focal else FOCAL_FALLBACK
    try:
        metric_depth, _ = run_depth_inference(img_bgr, focal_px)
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)},
        )

    # Normalize metric depth to 16-bit for PNG
    d_min, d_max = float(metric_depth.min()), float(metric_depth.max())
    if d_max - d_min > 1e-6:
        depth_16 = ((metric_depth - d_min) / (d_max - d_min) * 65535).astype(np.uint16)
    else:
        depth_16 = np.zeros_like(metric_depth, dtype=np.uint16)

    _, png_buf = cv2.imencode(".png", depth_16)
    return Response(content=png_buf.tobytes(), media_type="image/png")
