import { enforceBoneLengths, type LabPoseFrame } from './pose-adapters'
import type { PosePoint, TrackedPose } from './pose-retarget'

/** One Euro filter — low lag when moving, strong denoise when still. */
class OneEuroFilter {
  private xPrev?: number
  private dxPrev = 0
  private tPrev?: number
  private readonly minCutoff: number
  private readonly beta: number
  private readonly dCutoff: number

  constructor(minCutoff: number, beta: number, dCutoff: number) {
    this.minCutoff = minCutoff
    this.beta = beta
    this.dCutoff = dCutoff
  }

  filter(value: number, timestamp: number): number {
    if (this.tPrev === undefined || this.xPrev === undefined) {
      this.tPrev = timestamp
      this.xPrev = value
      this.dxPrev = 0
      return value
    }
    const dt = Math.max(1e-3, timestamp - this.tPrev)
    const dx = (value - this.xPrev) / dt
    const edx = lowPass(dx, this.dxPrev, alpha(dt, this.dCutoff))
    const cutoff = this.minCutoff + this.beta * Math.abs(edx)
    const x = lowPass(value, this.xPrev, alpha(dt, cutoff))
    this.tPrev = timestamp
    this.xPrev = x
    this.dxPrev = edx
    return x
  }
}

function alpha(dt: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * Math.max(1e-4, cutoff))
  return 1 / (1 + tau / dt)
}

function lowPass(value: number, previous: number, a: number): number {
  return previous + a * (value - previous)
}

type JointKey = Exclude<keyof TrackedPose, 'shoulderWidth'>

const JOINT_KEYS: JointKey[] = [
  'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
  'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
]

type JointFilters = Record<JointKey, { x: OneEuroFilter; y: OneEuroFilter; z: OneEuroFilter }>

function makeJointFilters(minCutoff: number, beta: number): JointFilters {
  const axis = () => new OneEuroFilter(minCutoff, beta, 1)
  return Object.fromEntries(
    JOINT_KEYS.map((key) => [key, { x: axis(), y: axis(), z: axis() }]),
  ) as JointFilters
}

function filterPoint(filters: JointFilters[JointKey], point: PosePoint, time: number): PosePoint {
  return {
    x: filters.x.filter(point.x, time),
    y: filters.y.filter(point.y, time),
    z: filters.z.filter(point.z, time),
  }
}

/**
 * Temporal denoise + optional bone-length snap on a TrackedPose sequence.
 * Designed for offline lab bake-off; same math works causal/online.
 */
export function smoothLabFrames(
  frames: LabPoseFrame[],
  options: { minCutoff?: number; beta?: number; boneLengths?: boolean } = {},
): LabPoseFrame[] {
  const minCutoff = options.minCutoff ?? 1.2
  const beta = options.beta ?? 0.02
  const boneLengths = options.boneLengths ?? true
  const filters = makeJointFilters(minCutoff, beta)

  return frames.map((frame) => {
    if (!frame.tracked) return frame
    const filtered: TrackedPose = {
      leftShoulder: filterPoint(filters.leftShoulder, frame.tracked.leftShoulder, frame.time),
      rightShoulder: filterPoint(filters.rightShoulder, frame.tracked.rightShoulder, frame.time),
      leftElbow: filterPoint(filters.leftElbow, frame.tracked.leftElbow, frame.time),
      rightElbow: filterPoint(filters.rightElbow, frame.tracked.rightElbow, frame.time),
      leftWrist: filterPoint(filters.leftWrist, frame.tracked.leftWrist, frame.time),
      rightWrist: filterPoint(filters.rightWrist, frame.tracked.rightWrist, frame.time),
      leftHip: filterPoint(filters.leftHip, frame.tracked.leftHip, frame.time),
      rightHip: filterPoint(filters.rightHip, frame.tracked.rightHip, frame.time),
      shoulderWidth: frame.tracked.shoulderWidth,
    }
    return {
      ...frame,
      tracked: boneLengths ? enforceBoneLengths(filtered) : filtered,
    }
  })
}
