import * as ort from 'onnxruntime-web'

export type WebPosePoint = { x: number; y: number; z: number; visibility: number }
export type WebPoseFrame = { time: number; landmarks: WebPosePoint[] | null }
type ProgressHandler = (detail: string, progress: number) => void

const MODEL_URLS = {
  rtmpose: '/assets/models/rtmpose-m.onnx',
  motionagformer: '/assets/models/motionagformer-xs.onnx',
} as const

const RTMPOSE_WIDTH = 192
const RTMPOSE_HEIGHT = 256
export const SAMPLE_FPS = 12
const AGFORMER_WINDOW = 27
const sessions: Partial<Record<keyof typeof MODEL_URLS, ort.InferenceSession>> = {}

type PersonBox = { x0: number; y0: number; x1: number; y1: number }

async function loadSession(name: keyof typeof MODEL_URLS): Promise<ort.InferenceSession> {
  const existing = sessions[name]
  if (existing) return existing
  const executionProviders = 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm']
  const session = await ort.InferenceSession.create(MODEL_URLS[name], { executionProviders })
  sessions[name] = session
  return session
}

function waitForSeek(video: HTMLVideoElement): Promise<void> {
  return new Promise((resolve) => video.addEventListener('seeked', () => resolve(), { once: true }))
}

async function seek(video: HTMLVideoElement, time: number): Promise<void> {
  if (Math.abs(video.currentTime - time) < .001) return
  const completed = waitForSeek(video)
  video.currentTime = time
  await completed
}

function getVideoDuration(video: HTMLVideoElement): number {
  if (Number.isFinite(video.duration) && video.duration > 0) return video.duration
  if (video.seekable.length) {
    const end = video.seekable.end(video.seekable.length - 1)
    const start = video.seekable.start(0)
    if (Number.isFinite(end) && end > start) return end - start
  }
  throw new Error('O video nao informa uma duracao finita. Recarregue a gravacao antes de processar.')
}

type InputGeometry = {
  tensor: ort.Tensor
  scale: number
  padX: number
  padY: number
  videoWidth: number
  videoHeight: number
  crop: PersonBox
}

function fullFrameBox(videoWidth: number, videoHeight: number): PersonBox {
  return { x0: 0, y0: 0, x1: videoWidth, y1: videoHeight }
}

/** Build a padded person bbox from COCO landmarks (normalized UV). */
export function bboxFromCoco(
  landmarks: readonly WebPosePoint[],
  videoWidth: number,
  videoHeight: number,
  padding = 0.4,
): PersonBox | null {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  let count = 0
  for (const point of landmarks) {
    if ((point.visibility ?? 1) < 0.12) continue
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue
    const x = point.x * videoWidth
    const y = point.y * videoHeight
    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x)
    maxY = Math.max(maxY, y)
    count += 1
  }
  if (count < 4) return null

  const width = Math.max(8, maxX - minX)
  const height = Math.max(8, maxY - minY)
  const padX = width * padding
  const padY = height * padding
  // Prefer a slightly tall box so arms/head stay inside after padding.
  const x0 = Math.max(0, minX - padX)
  const y0 = Math.max(0, minY - padY)
  const x1 = Math.min(videoWidth, maxX + padX)
  const y1 = Math.min(videoHeight, maxY + padY * 1.15)
  if (x1 - x0 < 16 || y1 - y0 < 16) return null
  return { x0, y0, x1, y1 }
}

function makeInput(video: HTMLVideoElement, canvas: HTMLCanvasElement, crop?: PersonBox): InputGeometry {
  const context = canvas.getContext('2d', { willReadFrequently: true })!
  const videoWidth = video.videoWidth
  const videoHeight = video.videoHeight
  const region = crop ?? fullFrameBox(videoWidth, videoHeight)
  const cropWidth = Math.max(1, region.x1 - region.x0)
  const cropHeight = Math.max(1, region.y1 - region.y0)
  // Letterbox the crop (or full frame) into 192x256 (RTMPose / MMPose top-down input).
  const scale = Math.min(RTMPOSE_WIDTH / cropWidth, RTMPOSE_HEIGHT / cropHeight)
  const drawWidth = cropWidth * scale
  const drawHeight = cropHeight * scale
  const padX = (RTMPOSE_WIDTH - drawWidth) / 2
  const padY = (RTMPOSE_HEIGHT - drawHeight) / 2
  context.fillStyle = `rgb(123, 116, 103)`
  context.fillRect(0, 0, RTMPOSE_WIDTH, RTMPOSE_HEIGHT)
  context.drawImage(
    video,
    region.x0, region.y0, cropWidth, cropHeight,
    padX, padY, drawWidth, drawHeight,
  )
  const pixels = context.getImageData(0, 0, RTMPOSE_WIDTH, RTMPOSE_HEIGHT).data
  const data = new Float32Array(3 * RTMPOSE_WIDTH * RTMPOSE_HEIGHT)
  const means = [123.675, 116.28, 103.53]
  const stds = [58.395, 57.12, 57.375]
  for (let index = 0; index < RTMPOSE_WIDTH * RTMPOSE_HEIGHT; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      data[channel * RTMPOSE_WIDTH * RTMPOSE_HEIGHT + index] = (pixels[index * 4 + channel] - means[channel]) / stds[channel]
    }
  }
  return {
    tensor: new ort.Tensor('float32', data, [1, 3, RTMPOSE_HEIGHT, RTMPOSE_WIDTH]),
    scale,
    padX,
    padY,
    videoWidth,
    videoHeight,
    crop: region,
  }
}

function argmaxSimcc(data: Float32Array, joint: number, bins: number): { index: number; confidence: number } {
  const offset = joint * bins
  let maxLogit = -Infinity
  let secondLogit = -Infinity
  let bestIndex = 0
  for (let index = 0; index < bins; index += 1) {
    const value = data[offset + index]
    if (value > maxLogit) {
      secondLogit = maxLogit
      maxLogit = value
      bestIndex = index
    } else if (value > secondLogit) {
      secondLogit = value
    }
  }
  // Softmax over hundreds of SimCC bins is ~0.002 even for good peaks — unusable as MediaPipe-like visibility.
  // Use peak margin vs the runner-up, mapped into roughly [0, 1].
  const margin = maxLogit - (Number.isFinite(secondLogit) ? secondLogit : maxLogit - 1)
  const confidence = 1 / (1 + Math.exp(-(margin * 4 - 1)))
  return { index: bestIndex, confidence }
}

function decodeRtmpose(outputs: Record<string, ort.Tensor>, geometry: InputGeometry): WebPosePoint[] {
  const simccX = outputs.simcc_x ?? Object.values(outputs)[0]
  const simccY = outputs.simcc_y ?? Object.values(outputs)[1]
  const xData = simccX.data as Float32Array
  const yData = simccY.data as Float32Array
  const xBins = simccX.dims[2]
  const yBins = simccY.dims[2]
  const points: WebPosePoint[] = []
  for (let joint = 0; joint < 17; joint += 1) {
    const x = argmaxSimcc(xData, joint, xBins)
    const y = argmaxSimcc(yData, joint, yBins)
    // SimCC split ratio = 2 → coordinate in the 192x256 model input space.
    const inputX = x.index / 2
    const inputY = y.index / 2
    const cropLocalX = (inputX - geometry.padX) / geometry.scale
    const cropLocalY = (inputY - geometry.padY) / geometry.scale
    const videoX = geometry.crop.x0 + cropLocalX
    const videoY = geometry.crop.y0 + cropLocalY
    points.push({
      x: videoX / geometry.videoWidth,
      y: videoY / geometry.videoHeight,
      z: 0,
      visibility: Math.min(x.confidence, y.confidence),
    })
  }
  return points
}

function h36mFromCoco(points: WebPosePoint[]): Float32Array {
  const coordinates = points.map((point) => [point.x, point.y])
  const midpoint = (first: number, second: number) => [
    (coordinates[first][0] + coordinates[second][0]) / 2,
    (coordinates[first][1] + coordinates[second][1]) / 2,
  ]
  const pelvis = midpoint(11, 12)
  const neck = midpoint(5, 6)
  const spine: number[] = [(pelvis[0] + neck[0]) / 2, (pelvis[1] + neck[1]) / 2]
  const head: number[] = [(coordinates[0][0] + neck[0]) / 2, (coordinates[0][1] + neck[1]) / 2]
  const map: number[][] = [
    pelvis, coordinates[12], coordinates[14], coordinates[16],
    coordinates[11], coordinates[13], coordinates[15],
    spine, neck, coordinates[0], head,
    coordinates[5], coordinates[7], coordinates[9],
    coordinates[6], coordinates[8], coordinates[10],
  ]
  const output = new Float32Array(17 * 3)
  map.forEach((point, index) => {
    output[index * 3] = point[0]
    output[index * 3 + 1] = point[1]
    output[index * 3 + 2] = points[Math.min(index, points.length - 1)].visibility
  })
  return output
}

function normalized3d(points: Float32Array): WebPosePoint[] {
  const rootX = points[0]
  const rootY = points[1]
  const rootZ = points[2]
  let scale = 1e-5
  for (let index = 0; index < 17; index += 1) {
    scale = Math.max(scale, Math.hypot(points[index * 3] - rootX, points[index * 3 + 1] - rootY, points[index * 3 + 2] - rootZ))
  }
  return Array.from({ length: 17 }, (_, index) => ({
    x: (points[index * 3] - rootX) / scale,
    y: (points[index * 3 + 1] - rootY) / scale,
    z: (points[index * 3 + 2] - rootZ) / scale,
    visibility: 1,
  }))
}

async function runMotionAgformer(sequence: Float32Array[], frames: WebPoseFrame[]): Promise<WebPoseFrame[]> {
  const session = await loadSession('motionagformer')
  const window = AGFORMER_WINDOW
  const output: WebPoseFrame[] = []
  for (let start = 0; start < sequence.length; start += window) {
    const chunk = sequence.slice(start, start + window)
    const data = new Float32Array(window * 17 * 3)
    for (let index = 0; index < window; index += 1) data.set(chunk[Math.min(index, chunk.length - 1)], index * 17 * 3)
    const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, [1, window, 17, 3]) })
    const poseTensor = Object.values(result)[0]
    const poseData = poseTensor.data as Float32Array
    for (let index = 0; index < chunk.length; index += 1) {
      output.push({
        time: frames[start + index].time,
        landmarks: normalized3d(poseData.slice(index * 17 * 3, (index + 1) * 17 * 3)),
      })
    }
  }
  return output
}

async function runRtmposePass(
  video: HTMLVideoElement,
  session: ort.InferenceSession,
  canvas: HTMLCanvasElement,
  times: number[],
  cropForIndex: (index: number, videoWidth: number, videoHeight: number) => PersonBox | undefined,
  onProgress: ProgressHandler,
  progressStart: number,
  progressEnd: number,
  label: string,
): Promise<WebPoseFrame[]> {
  const frames: WebPoseFrame[] = []
  for (let index = 0; index < times.length; index += 1) {
    await seek(video, times[index])
    const crop = cropForIndex(index, video.videoWidth, video.videoHeight)
    const geometry = makeInput(video, canvas, crop)
    const result = await session.run({ [session.inputNames[0]]: geometry.tensor })
    const landmarks = decodeRtmpose(result, geometry)
    frames.push({ time: video.currentTime, landmarks })
    const progress = progressStart + Math.round((index + 1) / times.length * (progressEnd - progressStart))
    onProgress(`${label}: amostra ${index + 1} de ${times.length}`, progress)
  }
  return frames
}

export type WebGpuPoseResult = {
  rtmpose: WebPoseFrame[]
  motionagformer: WebPoseFrame[]
  /** Second-pass RTMPose on person bbox derived from pass 1. */
  rtmposeCrop: WebPoseFrame[]
  /** AGFormer chunked on cropped 2D. */
  agformerCrop: WebPoseFrame[]
}

/** Sample video with RTMPose (+ crop pass), then lift with MotionAGFormer. */
export async function processWithWebGpu(video: HTMLVideoElement, onProgress: ProgressHandler): Promise<WebGpuPoseResult> {
  const duration = getVideoDuration(video)
  if (!video.videoWidth) throw new Error('Video invalido para processamento WebGPU.')
  onProgress('Carregando RTMPose no navegador', 5)
  const rtmpose = await loadSession('rtmpose')
  await loadSession('motionagformer')
  const canvas = document.createElement('canvas')
  canvas.width = RTMPOSE_WIDTH
  canvas.height = RTMPOSE_HEIGHT
  const originalTime = video.currentTime
  const times = getSampleTimes(duration)

  const frames = await runRtmposePass(
    video, rtmpose, canvas, times,
    () => undefined,
    onProgress, 8, 40,
    'RTMPose full-frame',
  )

  onProgress('RTMPose crop (bbox dos keypoints)', 42)
  const cropFrames = await runRtmposePass(
    video, rtmpose, canvas, times,
    (index, videoWidth, videoHeight) => {
      const landmarks = frames[index]?.landmarks
      if (!landmarks) return undefined
      return bboxFromCoco(landmarks, videoWidth, videoHeight) ?? undefined
    },
    onProgress, 42, 70,
    'RTMPose crop',
  )

  const sequence = frames.map((frame) => frame.landmarks
    ? h36mFromCoco(frame.landmarks)
    : new Float32Array(17 * 3))
  const cropSequence = cropFrames.map((frame) => frame.landmarks
    ? h36mFromCoco(frame.landmarks)
    : new Float32Array(17 * 3))

  onProgress('MotionAGFormer (full-frame 2D)', 74)
  const motionagformer = await runMotionAgformer(sequence, frames)
  onProgress('MotionAGFormer (crop 2D)', 90)
  const agformerCrop = await runMotionAgformer(cropSequence, cropFrames)

  await seek(video, originalTime)
  onProgress('Pipelines WebGPU concluidos', 100)
  return {
    rtmpose: frames,
    motionagformer,
    rtmposeCrop: cropFrames,
    agformerCrop,
  }
}

export async function seekVideo(video: HTMLVideoElement, time: number): Promise<void> {
  await seek(video, time)
}

export function getSampleTimes(duration: number): number[] {
  const sampleCount = Math.max(1, Math.ceil(duration * SAMPLE_FPS))
  return Array.from({ length: sampleCount }, (_, index) => Math.min(index / SAMPLE_FPS, Math.max(0, duration - .001)))
}
