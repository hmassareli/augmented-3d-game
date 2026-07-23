"""
Offline test script — processes a video file with DA3 metric depth
and plots the depth of the wrists over time.

Usage:
    python test_depth_video.py path/to/video.mp4

This helps verify that DA3 actually detects depth changes when
the arm extends (the core problem MediaPipe couldn't solve).

The script:
1. Runs MediaPipe Pose on each frame to get wrist (x,y) coordinates
2. Runs DA3 metric depth on each frame
3. Samples depth at the wrist positions
4. Plots depth over time + saves a CSV
5. Generates an annotated video with depth overlay
"""

from __future__ import annotations

import sys
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INFERENCE_SIZE = (518, 518)
FOCAL_PX = 800.0
OUTPUT_DIR = Path("depth_test_output")


def load_da3_model():
    """Load DA3 Metric-Large model."""
    from depth_anything_3.api import DepthAnything3
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DA3] Loading DA3METRIC-LARGE on {device}...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    print(f"[DA3] Loaded.")
    return model, device


def run_da3_depth(model, device, image_rgb):
    """Run DA3 inference, return metric depth in meters."""
    from PIL import Image as PILImage
    import torch

    pil = PILImage.fromarray(image_rgb)
    pil_resized = pil.resize((INFERENCE_SIZE[1], INFERENCE_SIZE[0]), PILImage.LANCZOS)

    with torch.no_grad():
        prediction = model.inference([pil_resized])

    raw_depth = prediction.depth[0]
    metric_depth = FOCAL_PX * raw_depth / 300.0
    return metric_depth, raw_depth


def sample_depth(depth_map, x_norm, y_norm):
    """Sample depth at normalized (0-1) coordinates with 5x5 median."""
    dh, dw = depth_map.shape
    px = int(x_norm * dw)
    py = int(y_norm * dh)
    px = max(0, min(dw - 1, px))
    py = max(0, min(dh - 1, py))
    x0, x1 = max(0, px - 2), min(dw, px + 3)
    y0, y1 = max(0, py - 2), min(dh, py + 3)
    return float(np.median(depth_map[y0:y1, x0:x1]))


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_depth_video.py path/to/video.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("[Init] Loading MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("[Init] Loading DA3...")
    da3_model, device = load_da3_model()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        str(OUTPUT_DIR / "depth_annotated.mp4"),
        fourcc,
        fps,
        (width, height + 200),  # extra space for depth plot
    )

    # Data storage
    frames_data = []
    frame_idx = 0

    print("[Processing] Starting frame-by-frame analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()

        # 1. MediaPipe Pose — get wrist positions
        results = pose.process(frame_rgb)
        left_wrist = None
        right_wrist = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_wrist = (lm[15].x, lm[15].y)  # left wrist
            right_wrist = (lm[16].x, lm[16].y)  # right wrist

        # 2. DA3 depth
        metric_depth, raw_depth = run_da3_depth(da3_model, device, frame_rgb)
        inference_ms = (time.perf_counter() - t0) * 1000

        # 3. Sample depth at wrists
        left_depth = sample_depth(metric_depth, *left_wrist) if left_wrist else None
        right_depth = sample_depth(metric_depth, *right_wrist) if right_wrist else None

        frames_data.append({
            "frame": frame_idx,
            "time_s": frame_idx / fps,
            "left_wrist_x": left_wrist[0] if left_wrist else "",
            "left_wrist_y": left_wrist[1] if left_wrist else "",
            "right_wrist_x": right_wrist[0] if right_wrist else "",
            "right_wrist_y": right_wrist[1] if right_wrist else "",
            "left_depth_m": left_depth,
            "right_depth_m": right_depth,
            "inference_ms": round(inference_ms, 1),
        })

        # 4. Annotated frame
        # Draw pose landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(66, 216, 174), thickness=2, circle_radius=3),
                mp.solutions.drawing_utils.DrawingSpec(color=(66, 216, 174), thickness=2),
            )

        # Draw depth values at wrists
        if left_wrist:
            lx, ly = int(left_wrist[0] * width), int(left_wrist[1] * height)
            cv2.putText(frame, f"L: {left_depth:.2f}m" if left_depth else "L: --",
                        (lx - 40, ly - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if right_wrist:
            rx, ry = int(right_wrist[0] * width), int(right_wrist[1] * height)
            cv2.putText(frame, f"R: {right_depth:.2f}m" if right_depth else "R: --",
                        (rx - 40, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Depth visualization strip at bottom
        depth_vis = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-6)
        depth_colored = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        depth_colored = cv2.resize(depth_colored, (width, 200))
        combined = np.vstack([frame, depth_colored])

        # Draw depth plot on the strip
        if len(frames_data) > 1:
            plot_h = 200
            plot_w = width
            plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
            plot[:] = (30, 30, 30)

            max_points = min(len(frames_data), 300)
            recent = frames_data[-max_points:]

            for i, d in enumerate(recent):
                x = int(i / max(max_points - 1, 1) * plot_w)
                if d["left_depth_m"] is not None and isinstance(d["left_depth_m"], float):
                    y = int(plot_h - (d["left_depth_m"] / 3.0) * plot_h)
                    y = max(0, min(plot_h - 1, y))
                    cv2.circle(plot, (x, y), 2, (0, 255, 0), -1)
                if d["right_depth_m"] is not None and isinstance(d["right_depth_m"], float):
                    y = int(plot_h - (d["right_depth_m"] / 3.0) * plot_h)
                    y = max(0, min(plot_h - 1, y))
                    cv2.circle(plot, (x, y), 2, (0, 100, 255), -1)

            cv2.putText(plot, "Left wrist depth (m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(plot, "Right wrist depth (m)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            cv2.putText(plot, f"Frame {frame_idx}/{total_frames}  Inf: {inference_ms:.0f}ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            combined = np.vstack([frame, plot])

        out_video.write(combined)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({inference_ms:.0f}ms) "
                  f"L={left_depth:.2f}m R={right_depth:.2f}m" if left_depth and right_depth
                  else f"  Frame {frame_idx}/{total_frames} ({inference_ms:.0f}ms)")

    cap.release()
    out_video.release()
    pose.close()

    # Save CSV
    csv_path = OUTPUT_DIR / "depth_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=frames_data[0].keys())
        writer.writeheader()
        writer.writerows(frames_data)

    print(f"\n[Done] Processed {frame_idx} frames")
    print(f"  Video: {OUTPUT_DIR / 'depth_annotated.mp4'}")
    print(f"  CSV:   {csv_path}")

    # Print summary
    left_depths = [d["left_depth_m"] for d in frames_data if d["left_depth_m"] is not None and isinstance(d["left_depth_m"], float)]
    right_depths = [d["right_depth_m"] for d in frames_data if d["right_depth_m"] is not None and isinstance(d["right_depth_m"], float)]

    if left_depths:
        print(f"\n  Left wrist depth:  min={min(left_depths):.2f}m  max={max(left_depths):.2f}m  delta={max(left_depths)-min(left_depths):.2f}m")
    if right_depths:
        print(f"  Right wrist depth: min={min(right_depths):.2f}m  max={max(right_depths):.2f}m  delta={max(right_depths)-min(right_depths):.2f}m")

    avg_inf = sum(d["inference_ms"] for d in frames_data) / len(frames_data)
    print(f"  Avg inference: {avg_inf:.0f}ms ({1000/avg_inf:.1f} FPS)")


if __name__ == "__main__":
    main()
