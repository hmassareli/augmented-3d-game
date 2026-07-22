import type { PosePoint, TrackedPose } from './pose-retarget'

export type Topology = 'blazepose' | 'coco' | 'h36m'

export type StickerLandmark = {
  x: number
  y: number
  z?: number
  visibility?: number
}

export type LabPoseFrame = {
  time: number
  sticker: { topology: Topology; landmarks: readonly StickerLandmark[] } | null
  tracked: TrackedPose | null
}

export const BLAZEPOSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
]

export const COCO_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
]

export const H36M_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
  [0, 7], [7, 8], [8, 9], [9, 10],
  [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16],
]

/**
 * MediaPipe world-ish contract (~meters, shoulder width ≈ 0.35), matched to Mixamo IK:
 * +X image-right (person's left when facing camera — same as BlazePose world in this lab),
 * Y increasing toward image-down relative lift (wrists below shoulders on screen → positive lift),
 * −Z toward camera / in front, +Z behind.
 */
const TARGET_SHOULDER_WIDTH = 0.35

function visibilityOk(point: StickerLandmark | undefined, min = 0.25): point is StickerLandmark {
  return Boolean(point) && (point!.visibility ?? 1) >= min
}

function shoulderWidthOf(left: PosePoint, right: PosePoint): number {
  return Math.max(0.05, Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z))
}

function scalePoint(point: PosePoint, factor: number): PosePoint {
  return { x: point.x * factor, y: point.y * factor, z: point.z * factor }
}

/**
 * Normalize any pipeline into the same metric contract the game IK expects
 * (MediaPipe world landmarks ≈ shoulder width 0.35 m).
 */
function calibrateTrackedPose(parts: Omit<TrackedPose, 'shoulderWidth'>): TrackedPose {
  const width = shoulderWidthOf(parts.leftShoulder, parts.rightShoulder)
  const factor = TARGET_SHOULDER_WIDTH / width
  const leftShoulder = scalePoint(parts.leftShoulder, factor)
  const rightShoulder = scalePoint(parts.rightShoulder, factor)
  const leftElbow = scalePoint(parts.leftElbow, factor)
  const rightElbow = scalePoint(parts.rightElbow, factor)
  const leftWrist = scalePoint(parts.leftWrist, factor)
  const rightWrist = scalePoint(parts.rightWrist, factor)
  let leftHip = scalePoint(parts.leftHip, factor)
  let rightHip = scalePoint(parts.rightHip, factor)

  // Keep hips under the shoulders on the same depth plane so noisy/missing
  // lower-body joints do not fold the spine forward in the Mixamo IK.
  const shoulderY = (leftShoulder.y + rightShoulder.y) / 2
  const shoulderZ = (leftShoulder.z + rightShoulder.z) / 2
  const hipY = Math.min(leftHip.y, rightHip.y, shoulderY - TARGET_SHOULDER_WIDTH * 0.95)
  leftHip = { x: leftShoulder.x, y: hipY, z: shoulderZ }
  rightHip = { x: rightShoulder.x, y: hipY, z: shoulderZ }

  return {
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    shoulderWidth: TARGET_SHOULDER_WIDTH,
  }
}

/** MediaPipe world landmarks → TrackedPose (same contract as the game). */
export function trackedFromBlazePoseWorld(
  world: readonly PosePoint[],
  image?: readonly StickerLandmark[],
): TrackedPose | null {
  const leftShoulder = world[11]
  const rightShoulder = world[12]
  const leftElbow = world[13]
  const rightElbow = world[14]
  const leftWrist = world[15]
  const rightWrist = world[16]
  const leftHip = world[23]
  const rightHip = world[24]
  if (!leftShoulder || !rightShoulder || !leftElbow || !rightElbow || !leftWrist || !rightWrist || !leftHip || !rightHip) {
    return null
  }
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x)
  if (shoulderWidth < 0.05) return null
  if (image) {
    const torso = Math.min(
      image[11]?.visibility ?? 0,
      image[12]?.visibility ?? 0,
      image[23]?.visibility ?? 0,
      image[24]?.visibility ?? 0,
    )
    if (torso < 0.35) return null
  }
  return calibrateTrackedPose({
    leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip,
  })
}

/**
 * Mild depth from foreshortening. Only when the segment is clearly compressed in-plane.
 * Image coords in; returns signed depth in the same units as `expectedLength`.
 * Negative = toward camera (in front), matching MediaPipe world Z.
 */
function mildDepth(
  from: { x: number; y: number },
  to: { x: number; y: number },
  expectedLength: number,
): number {
  const planar = Math.hypot(to.x - from.x, to.y - from.y)
  if (expectedLength < 1e-5 || planar / expectedLength > 0.88) return 0
  const depth = Math.sqrt(Math.max(0, expectedLength * expectedLength - planar * planar))
  // Prefer "in front of torso" when foreshortened — never invent behind-body depth.
  return -Math.min(depth, expectedLength * 0.45)
}

/**
 * COCO-17 image landmarks → MediaPipe-like world → TrackedPose.
 * Image: +x right, +y down.
 * Empirically MediaPipe world landmarks (lab audit) align with image axes for the Mixamo IK:
 * +X image-right (person's left when facing camera), +Y image-down-ish relative lift.
 * Do NOT use (0.5 − uv) — that mirrored X and inverted wrist lift vs MediaPipe, sending arms up.
 */
export function trackedFromCoco2d(points: readonly StickerLandmark[]): TrackedPose | null {
  if (!visibilityOk(points[5], 0.12) || !visibilityOk(points[6], 0.12)) return null

  const leftShoulder2d = points[5]
  const rightShoulder2d = points[6]
  const shoulderSpan = Math.hypot(leftShoulder2d.x - rightShoulder2d.x, leftShoulder2d.y - rightShoulder2d.y)
  if (shoulderSpan < 0.02 || shoulderSpan > 1.5) return null

  const metric = TARGET_SHOULDER_WIDTH / shoulderSpan
  const imageToWorld = (point: StickerLandmark, zImageUnits: number): PosePoint => ({
    x: (point.x - 0.5) * metric,
    y: (point.y - 0.5) * metric,
    z: zImageUnits * metric,
  })

  const upperArm = shoulderSpan * 0.9
  const foreArm = shoulderSpan * 0.82
  const leftElbow2d = visibilityOk(points[7], 0.1) ? points[7] : leftShoulder2d
  const rightElbow2d = visibilityOk(points[8], 0.1) ? points[8] : rightShoulder2d
  const leftWrist2d = visibilityOk(points[9], 0.1) ? points[9] : leftElbow2d
  const rightWrist2d = visibilityOk(points[10], 0.1) ? points[10] : rightElbow2d
  const leftHip2d = visibilityOk(points[11], 0.1)
    ? points[11]
    : { x: leftShoulder2d.x, y: Math.min(0.98, leftShoulder2d.y + shoulderSpan * 1.35), visibility: 0.4 }
  const rightHip2d = visibilityOk(points[12], 0.1)
    ? points[12]
    : { x: rightShoulder2d.x, y: Math.min(0.98, rightShoulder2d.y + shoulderSpan * 1.35), visibility: 0.4 }

  const leftElbowZ = mildDepth(leftShoulder2d, leftElbow2d, upperArm)
  const leftWristZ = leftElbowZ + mildDepth(leftElbow2d, leftWrist2d, foreArm)
  const rightElbowZ = mildDepth(rightShoulder2d, rightElbow2d, upperArm)
  const rightWristZ = rightElbowZ + mildDepth(rightElbow2d, rightWrist2d, foreArm)

  return calibrateTrackedPose({
    leftShoulder: imageToWorld(leftShoulder2d, 0),
    rightShoulder: imageToWorld(rightShoulder2d, 0),
    leftElbow: imageToWorld(leftElbow2d, leftElbowZ),
    rightElbow: imageToWorld(rightElbow2d, rightElbowZ),
    leftWrist: imageToWorld(leftWrist2d, leftWristZ),
    rightWrist: imageToWorld(rightWrist2d, rightWristZ),
    leftHip: imageToWorld(leftHip2d, 0),
    rightHip: imageToWorld(rightHip2d, 0),
  })
}

/**
 * H36M-17 root-relative 3D → MediaPipe-like world → TrackedPose.
 * MAG/MB is Y-down. Lab audit: keep Y-down (do not negate) so wrist lift matches MediaPipe
 * and Mixamo IK; X/Z already share MediaPipe's sign.
 */
export function trackedFromH36m3d(points: readonly StickerLandmark[]): TrackedPose | null {
  const leftShoulder = points[11]
  const rightShoulder = points[14]
  const leftElbow = points[12]
  const rightElbow = points[15]
  const leftWrist = points[13]
  const rightWrist = points[16]
  const leftHip = points[4]
  const rightHip = points[1]
  if (!leftShoulder || !rightShoulder || !leftElbow || !rightElbow || !leftWrist || !rightWrist || !leftHip || !rightHip) {
    return null
  }

  const toWorld = (point: StickerLandmark): PosePoint => ({
    x: point.x,
    y: point.y ?? 0,
    z: point.z ?? 0,
  })

  const leftS = toWorld(leftShoulder)
  const rightS = toWorld(rightShoulder)
  if (shoulderWidthOf(leftS, rightS) < 0.05) return null

  return calibrateTrackedPose({
    leftShoulder: leftS,
    rightShoulder: rightS,
    leftElbow: toWorld(leftElbow),
    rightElbow: toWorld(rightElbow),
    leftWrist: toWorld(leftWrist),
    rightWrist: toWorld(rightWrist),
    leftHip: toWorld(leftHip),
    rightHip: toWorld(rightHip),
  })
}

/**
 * Hybrid XY-lock: keep RTMPose/COCO image XY (accurate sticker), take relative Z from AGFormer H36M lift.
 * This is the transfer that matched ~0 UV error in sticker-vs-tracked debug.
 */
export function trackedFromHybrid(
  coco: readonly StickerLandmark[],
  h36m: readonly StickerLandmark[],
): TrackedPose | null {
  if (!visibilityOk(coco[5], 0.12) || !visibilityOk(coco[6], 0.12)) return null
  if (!h36m[11] || !h36m[14] || !h36m[12] || !h36m[15] || !h36m[13] || !h36m[16] || !h36m[4] || !h36m[1]) {
    return null
  }

  const leftShoulder2d = coco[5]
  const rightShoulder2d = coco[6]
  const shoulderSpan = Math.hypot(leftShoulder2d.x - rightShoulder2d.x, leftShoulder2d.y - rightShoulder2d.y)
  if (shoulderSpan < 0.02 || shoulderSpan > 1.5) return null

  const metric = TARGET_SHOULDER_WIDTH / shoulderSpan
  const leftElbow2d = visibilityOk(coco[7], 0.1) ? coco[7] : leftShoulder2d
  const rightElbow2d = visibilityOk(coco[8], 0.1) ? coco[8] : rightShoulder2d
  const leftWrist2d = visibilityOk(coco[9], 0.1) ? coco[9] : leftElbow2d
  const rightWrist2d = visibilityOk(coco[10], 0.1) ? coco[10] : rightElbow2d
  const leftHip2d = visibilityOk(coco[11], 0.1)
    ? coco[11]
    : { x: leftShoulder2d.x, y: Math.min(0.98, leftShoulder2d.y + shoulderSpan * 1.35), visibility: 0.4 }
  const rightHip2d = visibilityOk(coco[12], 0.1)
    ? coco[12]
    : { x: rightShoulder2d.x, y: Math.min(0.98, rightShoulder2d.y + shoulderSpan * 1.35), visibility: 0.4 }

  // Z only from H36M — keep native signs (same as trackedFromH36m3d).
  const toH = (point: StickerLandmark): PosePoint => ({
    x: point.x,
    y: point.y ?? 0,
    z: point.z ?? 0,
  })
  const hLeftShoulder = toH(h36m[11])
  const hRightShoulder = toH(h36m[14])
  const hWidth = shoulderWidthOf(hLeftShoulder, hRightShoulder)
  if (hWidth < 0.05) return null
  const zScale = TARGET_SHOULDER_WIDTH / hWidth
  const shoulderMidZ = (hLeftShoulder.z + hRightShoulder.z) / 2
  const depthAt = (point: StickerLandmark): number => (toH(point).z - shoulderMidZ) * zScale

  // Same image→world axes as trackedFromCoco2d (match MediaPipe / Mixamo IK).
  const xy = (point: StickerLandmark, z: number): PosePoint => ({
    x: (point.x - 0.5) * metric,
    y: (point.y - 0.5) * metric,
    z,
  })

  return calibrateTrackedPose({
    leftShoulder: xy(leftShoulder2d, 0),
    rightShoulder: xy(rightShoulder2d, 0),
    leftElbow: xy(leftElbow2d, depthAt(h36m[12])),
    rightElbow: xy(rightElbow2d, depthAt(h36m[15])),
    leftWrist: xy(leftWrist2d, depthAt(h36m[13])),
    rightWrist: xy(rightWrist2d, depthAt(h36m[16])),
    leftHip: xy(leftHip2d, depthAt(h36m[4])),
    rightHip: xy(rightHip2d, depthAt(h36m[1])),
  })
}

/** Snap arm segments to fixed lengths while preserving direction (stabilizes IK). */
export function enforceBoneLengths(pose: TrackedPose): TrackedPose {
  const upperArm = TARGET_SHOULDER_WIDTH * 0.9
  const foreArm = TARGET_SHOULDER_WIDTH * 0.82

  const place = (from: PosePoint, to: PosePoint, length: number): PosePoint => {
    const dx = to.x - from.x
    const dy = to.y - from.y
    const dz = to.z - from.z
    const current = Math.hypot(dx, dy, dz)
    // Degenerate → place slightly in front (−Z), not behind the torso.
    if (current < 1e-5) return { x: from.x, y: from.y, z: from.z - length }
    const scale = length / current
    return { x: from.x + dx * scale, y: from.y + dy * scale, z: from.z + dz * scale }
  }

  const leftElbow = place(pose.leftShoulder, pose.leftElbow, upperArm)
  const rightElbow = place(pose.rightShoulder, pose.rightElbow, upperArm)
  return {
    ...pose,
    leftElbow,
    rightElbow,
    leftWrist: place(leftElbow, pose.leftWrist, foreArm),
    rightWrist: place(rightElbow, pose.rightWrist, foreArm),
  }
}

/** Project H36M 3D into image-like coords for sticker overlay (model Y is down). */
export function projectH36mToSticker(points: readonly StickerLandmark[]): StickerLandmark[] {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const point of points) {
    minX = Math.min(minX, point.x)
    maxX = Math.max(maxX, point.x)
    minY = Math.min(minY, point.y ?? 0)
    maxY = Math.max(maxY, point.y ?? 0)
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-5)
  const scale = 0.55 / span
  const midX = (minX + maxX) / 2
  const midY = (minY + maxY) / 2
  return points.map((point) => ({
    x: 0.5 + (point.x - midX) * scale,
    y: 0.5 + ((point.y ?? 0) - midY) * scale,
    z: point.z,
    visibility: point.visibility ?? 1,
  }))
}

export function connectionsFor(topology: Topology): ReadonlyArray<readonly [number, number]> {
  if (topology === 'blazepose') return BLAZEPOSE_CONNECTIONS
  if (topology === 'h36m') return H36M_CONNECTIONS
  return COCO_CONNECTIONS
}
