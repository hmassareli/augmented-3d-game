"""Compatibility health endpoint for the browser-only pose comparison lab."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Counterpunch Pose Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):517[3-9]",
    allow_methods=["GET", "POST"],
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


@app.post("/api/normalize-video")
async def normalize_video(video: UploadFile = File(...), background_tasks: BackgroundTasks = None) -> FileResponse:
    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(status_code=415, detail="Envie um arquivo de video.")
    work_dir = Path(tempfile.mkdtemp(prefix="counterpunch-video-"))
    input_path = work_dir / "recording.webm"
    output_path = work_dir / "recording.mp4"
    with input_path.open("wb") as destination:
        shutil.copyfileobj(video.file, destination)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="O video gravado nao pode ser lido pelo decodificador local.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1
    capture.release()
    writer.release()
    if not frame_count or not output_path.exists() or not output_path.stat().st_size:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail="O video gravado nao contem frames decodificaveis.")
    cleanup = background_tasks or BackgroundTasks()
    cleanup.add_task(shutil.rmtree, work_dir, ignore_errors=True)
    return FileResponse(output_path, media_type="video/mp4", filename="recording.mp4", background=cleanup)
