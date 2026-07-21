"""Compatibility health endpoint for the browser-only pose comparison lab."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Counterpunch Pose Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):517[3-9]",
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ready": True,
        "mode": "browser-webgpu",
        "models": {
            "mediapipe": "browser",
            "rtmpose": "browser-webgpu",
            "motionbert": "browser-webgpu",
            "motionagformer": "browser-webgpu",
        },
    }
