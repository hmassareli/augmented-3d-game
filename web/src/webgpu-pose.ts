import * as ort from 'onnxruntime-web'

export type WebPosePoint = { x: number; y: number; z: number; visibility: number }
export type WebPoseFrame = { time: number; landmarks: WebPosePoint[] | null }
type ProgressHandler = (detail: string, progress: number) => void

const MODEL_URLS = {
  rtmpose: '/assets/models/rtmpose-m.onnx',
  motionbert: '/assets/models/motionbert-lite.onnx',
  motionagformer: '/assets/models/motionagformer-xs.onnx',
} as const

const RTMPOSE_WIDTH = 192
const RTMPOSE_HEIGHT = 256
const SAMPLE_FPS = 10
const sessions: Partial<Record<keyof typeof MODEL_URLS, ort.InferenceSession>> = {}

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

function makeInput(video: HTMLVideoElement, canvas: HTMLCanvasElement): { tensor: ort.Tensor; scaleX: number; scaleY: number; offsetX: number; offsetY: number; videoWidth: number; videoHeight: number } {
  const context = canvas.getContext('2d', { willReadFrequently: true })!
  const videoWidth = video.videoWidth
  const videoHeight = video.videoHeight
  const targetAspect = RTMPOSE_WIDTH / RTMPOSE_HEIGHT
  const padding = 1.25
  const sourceWidth = Math.max(videoWidth, videoHeight * targetAspect) * padding
  const sourceHeight = sourceWidth / targetAspect
  const offsetX = (videoWidth - sourceWidth) / 2
  const offsetY = (videoHeight - sourceHeight) / 2
  context.clearRect(0, 0, RTMPOSE_WIDTH, RTMPOSE_HEIGHT)
  context.drawImage(video, offsetX * RTMPOSE_WIDTH / sourceWidth, offsetY * RTMPOSE_HEIGHT / sourceHeight, videoWidth * RTMPOSE_WIDTH / sourceWidth, videoHeight * RTMPOSE_HEIGHT / sourceHeight)
  const pixels = context.getImageData(0, 0, RTMPOSE_WIDTH, RTMPOSE_HEIGHT).data
  const data = new Float32Array(3 * RTMPOSE_WIDTH * RTMPOSE_HEIGHT)
  const means = [123.675, 116.28, 103.53]
  const stds = [58.395, 57.12, 57.375]
  for (let index = 0; index < RTMPOSE_WIDTH * RTMPOSE_HEIGHT; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) data[channel * RTMPOSE_WIDTH * RTMPOSE_HEIGHT + index] = (pixels[index * 4 + channel] - means[channel]) / stds[channel]
  }
  return {
    tensor: new ort.Tensor('float32', data, [1, 3, RTMPOSE_HEIGHT, RTMPOSE_WIDTH]),
    scaleX: sourceWidth,
    scaleY: sourceHeight,
    offsetX,
    offsetY,
    videoWidth,
    videoHeight,
  }
}

function decodeRtmpose(outputs: Record<string, ort.Tensor>, geometry: ReturnType<typeof makeInput>): WebPosePoint[] {
  const [simccX, simccY] = Object.values(outputs)
  const xData = simccX.data as Float32Array
  const yData = simccY.data as Float32Array
  const xBins = simccX.dims[2]
  const yBins = simccY.dims[2]
  const points: WebPosePoint[] = []
  for (let joint = 0; joint < 17; joint += 1) {
    let xIndex = 0
    let yIndex = 0
    let xScore = -Infinity
    let yScore = -Infinity
    for (let index = 0; index < xBins; index += 1) if (xData[joint * xBins + index] > xScore) { xScore = xData[joint * xBins + index]; xIndex = index }
    for (let index = 0; index < yBins; index += 1) if (yData[joint * yBins + index] > yScore) { yScore = yData[joint * yBins + index]; yIndex = index }
    points.push({
      x: (xIndex / 2 / RTMPOSE_WIDTH * geometry.scaleX + geometry.offsetX) / geometry.videoWidth,
      y: (yIndex / 2 / RTMPOSE_HEIGHT * geometry.scaleY + geometry.offsetY) / geometry.videoHeight,
      z: 0,
      visibility: Math.min(xScore, yScore),
    })
  }
  return points
}

function h36mFromCoco(points: WebPosePoint[]): Float32Array {
  const coordinates = points.map((point) => [point.x, point.y])
  const midpoint = (first: number, second: number) => [(coordinates[first][0] + coordinates[second][0]) / 2, (coordinates[first][1] + coordinates[second][1]) / 2]
  const pelvis = midpoint(11, 12)
  const neck = midpoint(5, 6)
  const spine: number[] = [(pelvis[0] + neck[0]) / 2, (pelvis[1] + neck[1]) / 2]
  const head: number[] = [(coordinates[0][0] + neck[0]) / 2, (coordinates[0][1] + neck[1]) / 2]
  const map: number[][] = [pelvis, coordinates[12], coordinates[14], coordinates[16], coordinates[11], coordinates[13], coordinates[15], spine, neck, coordinates[0], head, coordinates[5], coordinates[7], coordinates[9], coordinates[6], coordinates[8], coordinates[10]]
  const output = new Float32Array(17 * 3)
  map.forEach((point, index) => { output[index * 3] = point[0]; output[index * 3 + 1] = point[1]; output[index * 3 + 2] = points[Math.min(index, points.length - 1)].visibility })
  return output
}

function normalized3d(points: Float32Array): WebPosePoint[] {
  const rootX = points[0]
  const rootY = points[1]
  const rootZ = points[2]
  let scale = 1e-5
  for (let index = 0; index < 17; index += 1) scale = Math.max(scale, Math.hypot(points[index * 3] - rootX, points[index * 3 + 1] - rootY, points[index * 3 + 2] - rootZ))
  return Array.from({ length: 17 }, (_, index) => ({ x: (points[index * 3] - rootX) / scale, y: (points[index * 3 + 1] - rootY) / scale, z: (points[index * 3 + 2] - rootZ) / scale, visibility: 1 }))
}

async function runTemporal(name: 'motionbert' | 'motionagformer', sequence: Float32Array[], frames: WebPoseFrame[]): Promise<WebPoseFrame[]> {
  const session = await loadSession(name)
  const window = name === 'motionbert' ? 243 : 27
  const output: WebPoseFrame[] = []
  for (let start = 0; start < sequence.length; start += window) {
    const chunk = sequence.slice(start, start + window)
    const data = new Float32Array(window * 17 * 3)
    for (let index = 0; index < window; index += 1) data.set(chunk[Math.min(index, chunk.length - 1)], index * 17 * 3)
    const result = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', data, [1, window, 17, 3]) })
    const poseData = Object.values(result)[0].data as Float32Array
    for (let index = 0; index < chunk.length; index += 1) output.push({ time: frames[start + index].time, landmarks: normalized3d(poseData.slice(index * 17 * 3, (index + 1) * 17 * 3)) })
  }
  return output
}

export async function processWithWebGpu(video: HTMLVideoElement, onProgress: ProgressHandler): Promise<Record<string, WebPoseFrame[]>> {
  const duration = getVideoDuration(video)
  if (!video.videoWidth) throw new Error('Video invalido para processamento WebGPU.')
  onProgress('Carregando RTMPose no navegador', 5)
  const rtmpose = await loadSession('rtmpose')
  const canvas = document.createElement('canvas')
  canvas.width = RTMPOSE_WIDTH
  canvas.height = RTMPOSE_HEIGHT
  const originalTime = video.currentTime
  const frames: WebPoseFrame[] = []
  const sequence: Float32Array[] = []
  const sampleCount = Math.max(1, Math.ceil(duration * SAMPLE_FPS))
  for (let index = 0; index < sampleCount; index += 1) {
    await seek(video, Math.min(index / SAMPLE_FPS, Math.max(0, duration - .001)))
    const geometry = makeInput(video, canvas)
    const result = await rtmpose.run({ [rtmpose.inputNames[0]]: geometry.tensor })
    const landmarks = decodeRtmpose(result, geometry)
    frames.push({ time: video.currentTime, landmarks })
    sequence.push(h36mFromCoco(landmarks))
    onProgress(`RTMPose WebGPU: amostra ${index + 1} de ${sampleCount}`, 5 + Math.round(index / sampleCount * 55))
  }
  await seek(video, originalTime)
  onProgress('MotionBERT WebGPU', 65)
  const motionbert = await runTemporal('motionbert', sequence, frames)
  onProgress('MotionAGFormer WebGPU', 82)
  const motionagformer = await runTemporal('motionagformer', sequence, frames)
  onProgress('Pipelines WebGPU concluidos', 100)
  return { rtmpose: frames, motionbert, motionagformer }
}