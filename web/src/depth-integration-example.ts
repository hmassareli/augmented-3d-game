/**
 * Integration patch for main.ts — shows how to wire DepthTracker
 * into the existing Counterpunch game loop.
 *
 * This is NOT a replacement for main.ts. It shows the key changes
 * needed to add DA3 depth tracking to the existing code.
 *
 * Search for "// DEPTH:" comments to see what was added.
 */

import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import {
  applyGuardPose,
  applyUpperBodyPose,
  cacheRetargetBones,
  orientBoneWithPalm,
  prepareAvatarRig,
  rotateBoneToward,
  type AvatarRig,
  type PosePoint,
  type TrackedPose,
} from "./pose-retarget";
// DEPTH: import the depth tracker
import { DepthTracker, type PunchEvent, type WristDepth } from "./depth-tracker";
import "./style.css";

// ... (existing UI setup code stays the same) ...

// DEPTH: Create depth tracker instance
const depthTracker = new DepthTracker("http://127.0.0.1:8001", 800);

// DEPTH: UI for depth toggle
// Add a button to the HTML to toggle depth tracking
const depthButton = document.createElement("button");
depthButton.id = "depth-button";
depthButton.textContent = "Depth: OFF";
depthButton.style.cssText = "position:absolute;bottom:12px;right:200px;padding:8px 16px;background:#1a3a3a;color:#42d8ae;border:1px solid #42d8ae;border-radius:6px;cursor:pointer;font-size:13px;";
document.querySelector("#app")!.appendChild(depthButton);

// DEPTH: Depth visualization overlay (shows depth map in corner)
const depthOverlay = document.createElement("img");
depthOverlay.id = "depth-overlay";
depthOverlay.style.cssText = "position:absolute;bottom:60px;right:12px;width:160px;height:120px;border:1px solid #42d8ae;border-radius:4px;display:none;pointer-events:none;";
document.querySelector("#app")!.appendChild(depthOverlay);

// DEPTH: Punch notification
const punchNotification = document.createElement("div");
punchNotification.id = "punch-notification";
punchNotification.style.cssText = "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:48px;font-weight:bold;color:#ff4444;text-shadow:0 0 20px #ff0000;opacity:0;transition:opacity 0.3s;pointer-events:none;";
document.querySelector("#app")!.appendChild(punchNotification);

// DEPTH: Wire up callbacks
depthTracker.onPunch = (event: PunchEvent) => {
  const power = (event.power * 100).toFixed(0);
  const label = event.type?.toUpperCase().replace("_", " ");
  punchNotification.textContent = `${label}! ${power}%`;
  punchNotification.style.opacity = "1";
  setTimeout(() => { punchNotification.style.opacity = "0"; }, 600);

  // Apply damage to opponent
  const damage = Math.round(event.power * 15);
  opponentStamina = Math.max(0, opponentStamina - damage);
  updateStaminaBars();

  // Trigger hit effects
  triggerHitEffect(event.type);
};

depthTracker.onDepthImage = (image: string) => {
  depthOverlay.src = image;
};

depthTracker.onDepthUpdate = (depths: WristDepth) => {
  // Can use depths.left and depths.right (in meters) to:
  // - Position gloves in Z space
  // - Adjust camera based on player distance
  // - Feed into IK for more realistic arm extension
};

// DEPTH: Toggle button
depthButton.addEventListener("click", async () => {
  if (depthTracker.isEnabled()) {
    depthTracker.setEnabled(false);
    depthButton.textContent = "Depth: OFF";
    depthOverlay.style.display = "none";
    return;
  }

  const healthy = await depthTracker.checkHealth();
  if (!healthy) {
    depthButton.textContent = "Backend offline";
    setTimeout(() => { depthButton.textContent = "Depth: OFF"; }, 2000);
    return;
  }

  depthTracker.setEnabled(true);
  depthButton.textContent = "Depth: ON";
  depthOverlay.style.display = "block";
});

// DEPTH: Track previous wrist positions for velocity calculation
let prevLeftWrist: { x: number; y: number } | null = null;
let prevRightWrist: { x: number; y: number } | null = null;
let prevPoseTime: number = 0;

// --- In the existing updateTrackedGloves() function, after getting pose results ---

// DEPTH: After MediaPipe gives us wrist positions, send to depth backend
// This goes inside the existing pose detection loop, after:
//   const result = poseLandmarker.detectForVideo(webcam, timestamp);

function updateDepthFromPose(
  landmarks: readonly PosePoint[] | undefined,
  timestamp: number,
) {
  if (!landmarks?.length || !depthTracker.isEnabled()) return;

  const leftWrist = landmarks[15];  // MediaPipe pose landmark 15 = left wrist
  const rightWrist = landmarks[16]; // MediaPipe pose landmark 16 = right wrist

  // Calculate 2D velocity (px/s) for punch classification
  const dt = prevPoseTime > 0 ? (timestamp - prevPoseTime) / 1000 : 0.016;
  let leftVel = 0;
  let rightVel = 0;

  if (leftWrist && prevLeftWrist) {
    leftVel = Math.hypot(
      leftWrist.x - prevLeftWrist.x,
      leftWrist.y - prevLeftWrist.y,
    ) / dt;
  }
  if (rightWrist && prevRightWrist) {
    rightVel = Math.hypot(
      rightWrist.x - prevRightWrist.x,
      rightWrist.y - prevRightWrist.y,
    ) / dt;
  }

  // Send frame to depth backend (async, non-blocking)
  depthTracker.processFrame(
    webcam,
    leftWrist ? { x: leftWrist.x, y: leftWrist.y } : null,
    rightWrist ? { x: rightWrist.x, y: rightWrist.y } : null,
    timestamp,
    leftVel,
    rightVel,
  );

  prevLeftWrist = leftWrist ? { x: leftWrist.x, y: leftWrist.y } : null;
  prevRightWrist = rightWrist ? { x: rightWrist.x, y: rightWrist.y } : null;
  prevPoseTime = timestamp;
}

// --- In the existing render loop (requestAnimationFrame) ---

// DEPTH: Call depth update inside the main loop, after pose detection
// The existing code already calls updateTrackedGloves() in the loop.
// Add updateDepthFromPose() right after it:

function gameLoop() {
  // ... existing code ...
  updateTrackedGloves();
  // DEPTH: feed pose data to depth tracker
  if (latestPose) {
    updateDepthFromPose(latestPose.landmarks, performance.now());
  }
  // ... rest of existing code ...
  requestAnimationFrame(gameLoop);
}

// --- Helper: trigger hit effects ---

function triggerHitEffect(punchType: PunchEvent["type"]) {
  // Camera shake
  cameraOrbitYaw += (Math.random() - 0.5) * 0.08;

  // Flash the canvas
  canvas.style.transition = "filter 0.05s";
  canvas.style.filter = "brightness(1.5)";
  setTimeout(() => { canvas.style.filter = "brightness(1)"; }, 80);

  // Play sound (if audio system exists)
  // audio.playImpact(punchType);
}

// --- Helper: update stamina bars ---

function updateStaminaBars() {
  const playerBar = document.querySelector("#player-stamina");
  const opponentBar = document.querySelector("#opponent-stamina");
  if (playerBar) (playerBar as HTMLElement).style.width = `${playerStamina}%`;
  if (opponentBar) (opponentBar as HTMLElement).style.width = `${opponentStamina}%`;
}
