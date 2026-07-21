import './comparison.css'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { clone } from 'three/addons/utils/SkeletonUtils.js'
import { ArrayBufferTarget, Muxer } from 'mp4-muxer'
import { processWithWebGpu } from './webgpu-pose'

declare const MediaStreamTrackProcessor: new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <main class="lab-shell">
    <header class="lab-header">
      <a class="brand" href="/" aria-label="Voltar ao Counterpunch">COUNTERPUNCH <span>LAB</span></a>
      <p>COMPARADOR DE RASTREAMENTO 3D</p>
      <div class="run-status"><i></i><span id="run-status">aguardando video</span></div>
    </header>

    <section class="source-zone" aria-labelledby="source-title">
      <div class="source-copy">
        <p class="section-kicker">FONTE</p>
        <h1 id="source-title">Movimento em uma tomada.</h1>
        <p>Carregue um video curto de corpo inteiro para reproduzir a estimativa de cada pipeline no mesmo tempo.</p>
        <div class="source-actions">
          <label class="file-button" for="video-input">Carregar video</label>
          <button id="record-button" class="record-button" type="button">Gravar camera</button>
          <button id="stop-recording-button" class="stop-recording-button" type="button" disabled>Parar</button>
          <button id="process-button" class="process-button" type="button" disabled>Processar comparacao</button>
        </div>
        <input id="video-input" type="file" accept="video/*" hidden />
        <p id="file-name" class="file-name">Nenhum arquivo selecionado</p>
        <section id="processing-status" class="processing-status" aria-live="polite" hidden>
          <div><span id="processing-stage">Na fila</span><strong id="processing-percent">0%</strong></div>
          <progress id="processing-progress" max="100" value="0">0%</progress>
          <p id="processing-detail">O video sera analisado por um pipeline de cada vez.</p>
        </section>
      </div>
      <div class="video-frame">
        <video id="source-video" playsinline muted></video>
        <div id="video-empty" class="video-empty"><span>VIDEO TESTE</span></div>
        <div class="video-controls">
          <button id="play-button" type="button" aria-label="Reproduzir video" disabled>▶</button>
          <input id="timeline" type="range" min="0" max="1" value="0" step="0.001" aria-label="Linha do tempo" disabled />
          <output id="time-output">00:00 / 00:00</output>
        </div>
      </div>
    </section>

    <section class="compare-toolbar" aria-label="Controles da comparacao">
      <div class="model-count"><strong>04</strong><span>pipelines</span></div>
      <div class="view-toggle" role="group" aria-label="Visualizacao dos resultados">
        <button type="button" class="is-active" data-view="avatar">MODELO 3D</button>
        <button type="button" data-view="skeleton">STICKER</button>
      </div>
      <p id="pipeline-note" class="pipeline-note">Verificando backend de inferencia...</p>
    </section>

    <section id="model-grid" class="model-grid" aria-label="Resultados dos modelos"></section>
  </main>
`

const labShell = document.querySelector<HTMLElement>('.lab-shell')!

const modelDefinitions = [
  { name: 'MediaPipe', score: '★★★', color: '#5cd5bf', pipeline: 'mediapipe' },
  { name: 'RTMPose', score: '★★★★', color: '#e6bd68', pipeline: 'rtmpose' },
  { name: 'RTMPose + MotionBERT', score: '★★★★★', color: '#7da9f5', pipeline: 'motionbert' },
  { name: 'RTMPose + MotionAGFormer', score: '★★★★★+', color: '#cb91dc', pipeline: 'motionagformer' },
]

const modelGrid = document.querySelector<HTMLDivElement>('#model-grid')!
modelGrid.innerHTML = modelDefinitions.map((model, index) => `
  <article class="model-card" style="--accent: ${model.color}; --delay: ${index * 0.13}s">
    <header>
      <div><p>INFERENCIA REAL</p><h2>${model.name}</h2></div>
      <strong>${model.score}</strong>
    </header>
    <div class="avatar-stage">
      <canvas class="avatar-canvas" data-model-index="${index}" aria-label="Avatar 3D ${model.name}"></canvas>
      <canvas class="pose-canvas" data-model-index="${index}" aria-label="Sticker ${model.name}"></canvas>
    </div>
    <footer><span class="model-state">aguardando video</span><span class="frame-rate">-- FPS</span></footer>
  </article>
`).join('')

const videoInput = document.querySelector<HTMLInputElement>('#video-input')!
const sourceVideo = document.querySelector<HTMLVideoElement>('#source-video')!
const videoEmpty = document.querySelector<HTMLDivElement>('#video-empty')!
const playButton = document.querySelector<HTMLButtonElement>('#play-button')!
const timeline = document.querySelector<HTMLInputElement>('#timeline')!
const timeOutput = document.querySelector<HTMLOutputElement>('#time-output')!
const fileName = document.querySelector<HTMLElement>('#file-name')!
const runStatus = document.querySelector<HTMLElement>('#run-status')!
const pipelineNote = document.querySelector<HTMLElement>('#pipeline-note')!
const recordButton = document.querySelector<HTMLButtonElement>('#record-button')!
const stopRecordingButton = document.querySelector<HTMLButtonElement>('#stop-recording-button')!
const processButton = document.querySelector<HTMLButtonElement>('#process-button')!
const processingStatus = document.querySelector<HTMLElement>('#processing-status')!
const processingStage = document.querySelector<HTMLElement>('#processing-stage')!
const processingPercent = document.querySelector<HTMLElement>('#processing-percent')!
const processingProgress = document.querySelector<HTMLProgressElement>('#processing-progress')!
const processingDetail = document.querySelector<HTMLElement>('#processing-detail')!
const canvasElements = [...document.querySelectorAll<HTMLCanvasElement>('.pose-canvas')]
const avatarCanvases = [...document.querySelectorAll<HTMLCanvasElement>('.avatar-canvas')]
let videoUrl: string | undefined
let view: 'avatar' | 'skeleton' = 'avatar'
let poseLandmarker: PoseLandmarker | undefined
let modelLoading: Promise<void> | undefined
let lastInferenceTime = -1
type Landmark = { x: number; y: number; visibility?: number }
let mediaPipeLandmarks: readonly Landmark[] | undefined
let cameraStream: MediaStream | undefined
type CameraRecorder = { stop: () => Promise<Blob> }
let cameraRecorder: CameraRecorder | undefined
let sourceBlob: Blob | undefined
let sourceDuration: number | undefined
let recordingStartedAt = 0
let comparisonReady = false
type PoseFrame = { time: number; landmarks: readonly Landmark[] | null }
let backendFrames: Partial<Record<string, PoseFrame[]>> = {}

type ProcessJob = {
  status: 'processing' | 'complete' | 'failed'
  stage: string
  progress: number
  detail: string
}

type AvatarPreview = { renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; avatar?: THREE.Group; mixer?: THREE.AnimationMixer }
const avatarPreviews: AvatarPreview[] = []
const avatarLoader = new FBXLoader()
let avatarTemplate: THREE.Group | undefined

function formatTime(seconds: number): string {
  const wholeSeconds = Number.isFinite(seconds) ? Math.floor(seconds) : 0
  return `${String(Math.floor(wholeSeconds / 60)).padStart(2, '0')}:${String(wholeSeconds % 60).padStart(2, '0')}`
}

function effectiveDuration(): number {
  return Number.isFinite(sourceVideo.duration) && sourceVideo.duration > 0 ? sourceVideo.duration : sourceDuration ?? 0
}

function updateTimeline(): void {
  timeline.value = String(sourceVideo.currentTime)
  timeOutput.value = `${formatTime(sourceVideo.currentTime)} / ${formatTime(effectiveDuration())}`
}

function createAvatarPreview(canvas: HTMLCanvasElement): AvatarPreview {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  const scene = new THREE.Scene()
  scene.add(new THREE.HemisphereLight('#d6f4ee', '#182022', 2.7))
  const light = new THREE.DirectionalLight('#ffe7c2', 3)
  light.position.set(2, 4, 4)
  scene.add(light)
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(1.1, 40),
    new THREE.MeshBasicMaterial({ color: '#243132', transparent: true, opacity: .62 }),
  )
  floor.rotation.x = -Math.PI / 2
  scene.add(floor)
  const camera = new THREE.PerspectiveCamera(31, 1, .1, 20)
  camera.position.set(0, 1.12, 3.35)
  camera.lookAt(0, .82, 0)
  return { renderer, scene, camera }
}

function cacheRestPose(avatar: THREE.Group): void {
  avatar.traverse((object) => {
    if (object instanceof THREE.Bone) object.userData.restQuaternion = object.quaternion.clone()
    if (object instanceof THREE.Mesh) {
      object.castShadow = true
      object.receiveShadow = true
    }
  })
}

async function loadAvatars(): Promise<void> {
  if (avatarTemplate) return
  avatarTemplate = await avatarLoader.loadAsync('/assets/mixamo/character/y-bot.fbx')
  avatarTemplate.scale.setScalar(.01)
  cacheRestPose(avatarTemplate)
  avatarPreviews.forEach((preview, index) => {
    const avatar = clone(avatarTemplate!) as THREE.Group
    avatar.position.y = 0
    avatar.rotation.y = Math.PI
    avatar.userData.phase = index * .64
    preview.avatar = avatar
    preview.scene.add(avatar)
  })
}

function resetAvatarPose(avatar: THREE.Group): void {
  avatar.traverse((object) => {
    if (object instanceof THREE.Bone && object.userData.restQuaternion instanceof THREE.Quaternion) {
      object.quaternion.copy(object.userData.restQuaternion)
    }
  })
}

function rotateArm(avatar: THREE.Group, side: 'Left' | 'Right', swing: number, elbow: number): void {
  const arm = avatar.getObjectByName(`mixamorig${side}Arm`)
  const forearm = avatar.getObjectByName(`mixamorig${side}ForeArm`)
  const direction = side === 'Left' ? -1 : 1
  if (arm instanceof THREE.Bone) arm.rotation.z += direction * swing
  if (forearm instanceof THREE.Bone) forearm.rotation.z += direction * elbow
}

function rotateLeg(avatar: THREE.Group, side: 'Left' | 'Right', swing: number): void {
  const leg = avatar.getObjectByName(`mixamorig${side}UpLeg`)
  const direction = side === 'Left' ? -1 : 1
  if (leg instanceof THREE.Bone) leg.rotation.x += direction * swing
}

function landmarksFor(index: number): readonly Landmark[] | undefined {
  const frames = backendFrames[modelDefinitions[index].pipeline]
  if (frames?.length) {
    const nearest = frames.reduce((current, frame) => Math.abs(frame.time - sourceVideo.currentTime) < Math.abs(current.time - sourceVideo.currentTime) ? frame : current)
    return nearest.landmarks ?? undefined
  }
  return index === 0 ? mediaPipeLandmarks : undefined
}

type PosePoint = { x: number; y: number; visibility: number }

function posePoints(index: number, landmarks: readonly Landmark[]): PosePoint[] {
  if (index < 2) return landmarks.map((landmark) => ({ x: landmark.x, y: landmark.y, visibility: landmark.visibility ?? 1 }))

  const visible = landmarks.filter((landmark) => (landmark.visibility ?? 1) >= .35)
  if (!visible.length) return []
  const minX = Math.min(...visible.map((landmark) => landmark.x))
  const maxX = Math.max(...visible.map((landmark) => landmark.x))
  const minY = Math.min(...visible.map((landmark) => landmark.y))
  const maxY = Math.max(...visible.map((landmark) => landmark.y))
  const centerX = (minX + maxX) / 2
  const centerY = (minY + maxY) / 2
  const scale = Math.max(maxX - minX, maxY - minY, .01)
  return landmarks.map((landmark) => ({
    x: .5 + (landmark.x - centerX) / scale * .34,
    y: .56 - (landmark.y - centerY) / scale * .34,
    visibility: landmark.visibility ?? 1,
  }))
}

function poseAvatar(avatar: THREE.Group, index: number, _time: number): void {
  avatar.rotation.y = Math.PI
  resetAvatarPose(avatar)
  const landmarks = landmarksFor(index)
  if (landmarks) {
    const points = posePoints(index, landmarks)
    const spine = avatar.getObjectByName('mixamorigSpine')
    const isMediaPipe = index === 0
    const leftShoulder = points[isMediaPipe ? 11 : 5]
    const leftElbow = points[isMediaPipe ? 13 : 7]
    const leftWrist = points[isMediaPipe ? 15 : 9]
    const rightShoulder = points[isMediaPipe ? 12 : 6]
    const rightElbow = points[isMediaPipe ? 14 : 8]
    const rightWrist = points[isMediaPipe ? 16 : 10]
    const leftHip = points[isMediaPipe ? 23 : 11]
    const leftKnee = points[isMediaPipe ? 25 : 13]
    const rightHip = points[isMediaPipe ? 24 : 12]
    const rightKnee = points[isMediaPipe ? 26 : 14]
    if (spine instanceof THREE.Bone && leftShoulder && rightShoulder) spine.rotation.y = (rightShoulder.x - leftShoulder.x) * .7
    if (leftShoulder && leftElbow) rotateArm(avatar, 'Left', (leftElbow.y - leftShoulder.y) * 6, leftWrist ? (leftWrist.y - leftElbow.y) * 5 : 0)
    if (rightShoulder && rightElbow) rotateArm(avatar, 'Right', (rightElbow.y - rightShoulder.y) * 6, rightWrist ? (rightWrist.y - rightElbow.y) * 5 : 0)
    if (leftHip && leftKnee) rotateLeg(avatar, 'Left', (leftKnee.y - leftHip.y - .2) * 3)
    if (rightHip && rightKnee) rotateLeg(avatar, 'Right', (rightKnee.y - rightHip.y - .2) * 3)
  }
}

function renderAvatars(time: number): void {
  for (const preview of avatarPreviews) {
    const canvas = preview.renderer.domElement
    const bounds = canvas.getBoundingClientRect()
    const width = Math.max(1, Math.round(bounds.width * devicePixelRatio))
    const height = Math.max(1, Math.round(bounds.height * devicePixelRatio))
    if (canvas.width !== width || canvas.height !== height) {
      preview.renderer.setSize(bounds.width, bounds.height, false)
      preview.camera.aspect = bounds.width / bounds.height
      preview.camera.updateProjectionMatrix()
    }
    if (preview.avatar) poseAvatar(preview.avatar, avatarPreviews.indexOf(preview), time)
    preview.renderer.render(preview.scene, preview.camera)
  }
}

function clearCameraStream(): void {
  cameraStream?.getTracks().forEach((track) => track.stop())
  cameraStream = undefined
}

function createCameraRecorder(stream: MediaStream): CameraRecorder {
  const track = stream.getVideoTracks()[0]
  if (!track) throw new Error('A camera nao forneceu uma faixa de video.')
  const settings = track.getSettings()
  const width = settings.width || 1280
  const height = settings.height || 720
  const target = new ArrayBufferTarget()
  const muxer = new Muxer({
    target,
    video: { codec: 'avc', width, height },
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  })
  const encoder = new VideoEncoder({
    output: (chunk, metadata) => muxer.addVideoChunk(chunk, metadata),
    error: (error) => console.error('Camera MP4 encoder failed', error),
  })
  encoder.configure({
    codec: 'avc1.42001f',
    width,
    height,
    bitrate: 3_000_000,
    framerate: settings.frameRate || 30,
    avc: { format: 'avc' },
  })
  const processor = new MediaStreamTrackProcessor({ track })
  const reader = processor.readable.getReader()
  let stopping = false
  let frameCount = 0
  const pump = (async () => {
    while (!stopping) {
      const { done, value } = await reader.read()
      if (done) break
      encoder.encode(value, { keyFrame: frameCount++ % 60 === 0 })
      value.close()
    }
  })()
  return {
    stop: async () => {
      stopping = true
      await reader.cancel()
      await pump.catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return
        throw error
      })
      await encoder.flush()
      muxer.finalize()
      return new Blob([target.buffer], { type: 'video/mp4' })
    },
  }
}

function loadVideoSource(url: string, label: string, blob: Blob, duration?: number): void {
  clearCameraStream()
  if (videoUrl) URL.revokeObjectURL(videoUrl)
  videoUrl = url
  lastInferenceTime = -1
  mediaPipeLandmarks = undefined
  backendFrames = {}
  comparisonReady = false
  processingStatus.hidden = true
  sourceBlob = blob
  sourceDuration = duration
  timeline.disabled = true
  playButton.disabled = true
  processButton.disabled = true
  sourceVideo.srcObject = null
  sourceVideo.src = url
  sourceVideo.load()
  fileName.textContent = label
}

function updateProcessingStatus(job: ProcessJob): void {
  processingStatus.hidden = false
  processingStage.textContent = job.stage === 'queued' ? 'Na fila' : job.stage
  processingPercent.textContent = `${job.progress}%`
  processingProgress.value = job.progress
  processingDetail.textContent = job.detail
  pipelineNote.textContent = job.detail
  document.querySelectorAll<HTMLElement>('.model-card').forEach((card, index) => {
    const isActive = modelDefinitions[index].pipeline === job.stage
    card.classList.toggle('is-processing', isActive)
    if (isActive) card.querySelector<HTMLElement>('.model-state')!.textContent = 'processando agora'
  })
}

async function startRecording(): Promise<void> {
  if (!navigator.mediaDevices?.getUserMedia || !('VideoEncoder' in window) || !('MediaStreamTrackProcessor' in window)) {
    runStatus.textContent = 'gravacao indisponivel neste navegador'
    return
  }
  try {
    comparisonReady = false
    runStatus.textContent = 'carregando MediaPipe'
    await loadMediaPipe()
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
    sourceVideo.src = ''
    sourceVideo.srcObject = cameraStream
    sourceVideo.muted = true
    await sourceVideo.play()
    videoEmpty.hidden = true
    cameraRecorder = createCameraRecorder(cameraStream)
    recordingStartedAt = performance.now()
    recordButton.disabled = true
    stopRecordingButton.disabled = false
    runStatus.textContent = 'gravando camera'
    fileName.textContent = 'gravando agora...'
  } catch (error) {
    console.error('Unable to record camera', error)
    clearCameraStream()
    runStatus.textContent = 'acesso a camera negado'
  }
}

async function stopRecording(): Promise<void> {
  if (!cameraRecorder) return
  stopRecordingButton.disabled = true
  runStatus.textContent = 'finalizando MP4'
  try {
    const recording = await cameraRecorder.stop()
    const duration = Math.max(.1, (performance.now() - recordingStartedAt) / 1000)
    cameraRecorder = undefined
    loadVideoSource(URL.createObjectURL(recording), `gravacao-${new Date().toLocaleTimeString('pt-BR').replaceAll(':', '-')}.mp4`, recording, duration)
    runStatus.textContent = 'validando gravacao MP4'
  } catch (error) {
    console.error('Unable to finalize camera recording', error)
    runStatus.textContent = 'falha ao finalizar gravacao'
    recordButton.disabled = false
    stopRecordingButton.disabled = true
  }
}

async function processWithWebGpuOnly(): Promise<void> {
  if (!sourceBlob) return
  processButton.disabled = true
  runStatus.textContent = 'iniciando WebGPU'
  updateProcessingStatus({ status: 'processing', stage: 'rtmpose', progress: 0, detail: 'Preparando modelos WebGPU no navegador' })
  try {
    const webGpuPipelines = await processWithWebGpu(sourceVideo, (detail, progress) => {
      const stage = detail.startsWith('MotionBERT') ? 'motionbert' : detail.startsWith('MotionAGFormer') ? 'motionagformer' : 'rtmpose'
      updateProcessingStatus({ status: 'processing', stage, progress, detail })
    }, sourceDuration)
    backendFrames = {
      mediapipe: [],
      rtmpose: webGpuPipelines.rtmpose,
      motionbert: webGpuPipelines.motionbert,
      motionagformer: webGpuPipelines.motionagformer,
    }
    document.querySelectorAll<HTMLElement>('.model-card').forEach((card, index) => {
      const frames = backendFrames[modelDefinitions[index].pipeline]
      const state = card.querySelector<HTMLElement>('.model-state')
      const frameRate = card.querySelector<HTMLElement>('.frame-rate')
      if (state) state.textContent = index === 0 ? 'MediaPipe local no navegador' : frames?.length ? 'processado via WebGPU' : 'sem pessoa detectada'
      if (frameRate) frameRate.textContent = 'WebGPU'
    })
    document.querySelectorAll<HTMLElement>('.model-card').forEach((card) => card.classList.remove('is-processing'))
    updateProcessingStatus({ status: 'complete', stage: 'complete', progress: 100, detail: 'Quatro pipelines WebGPU concluidos' })
    runStatus.textContent = 'comparacao WebGPU pronta'
    comparisonReady = true
    renderPoses()
  } catch (error) {
    console.error('Unable to process video with WebGPU', error)
    runStatus.textContent = 'falha no processamento WebGPU'
    processingDetail.textContent = error instanceof Error ? error.message : 'Falha desconhecida durante o processamento.'
  } finally {
    processButton.disabled = false
  }
}

async function checkBackend(): Promise<void> {
  const webGpuAvailable = 'gpu' in navigator && Boolean(await navigator.gpu?.requestAdapter())
  pipelineNote.textContent = webGpuAvailable
    ? 'WebGPU ativo: RTMPose, MotionBERT e MotionAGFormer executam no navegador.'
    : 'WebGPU indisponivel: os modelos usarao o fallback WASM no navegador.'
  document.querySelectorAll<HTMLElement>('.model-state').forEach((state) => { state.textContent = 'pronto no navegador' })
}

async function loadMediaPipe(): Promise<void> {
  if (poseLandmarker) return
  if (modelLoading) return modelLoading
  modelLoading = (async () => {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
    )
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    })
  })()
  try {
    await modelLoading
  } finally {
    modelLoading = undefined
  }
}

function detectMediaPipePose(): void {
  const frames = backendFrames.mediapipe
  if (frames?.length) {
    const nearest = frames.reduce((current, frame) => Math.abs(frame.time - sourceVideo.currentTime) < Math.abs(current.time - sourceVideo.currentTime) ? frame : current)
    mediaPipeLandmarks = nearest.landmarks ?? undefined
    return
  }
  if (!poseLandmarker || sourceVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
  if (sourceVideo.currentTime === lastInferenceTime) return
  lastInferenceTime = sourceVideo.currentTime
  mediaPipeLandmarks = poseLandmarker.detectForVideo(sourceVideo, performance.now()).landmarks[0] as readonly Landmark[] | undefined
}

function drawPlaceholder(canvas: HTMLCanvasElement, index: number, time: number): void {
  const bounds = canvas.getBoundingClientRect()
  const width = Math.max(1, Math.round(bounds.width * devicePixelRatio))
  const height = Math.max(1, Math.round(bounds.height * devicePixelRatio))
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  const context = canvas.getContext('2d')!
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
  context.clearRect(0, 0, bounds.width, bounds.height)
  const accent = modelDefinitions[index].color
  const centerX = bounds.width / 2
  const centerY = bounds.height * 0.56
  const pace = time * (2.8 + index * 0.07)
  const sway = Math.sin(pace + index) * 10
  const punch = Math.max(0, Math.sin(pace * 0.67 + index * 0.8)) * (18 + index * 3)

  context.strokeStyle = accent
  context.fillStyle = accent
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = view === 'skeleton' ? 3 : 5
  context.shadowColor = `${accent}55`
  context.shadowBlur = 16

  const headY = centerY - 69
  const shoulderY = centerY - 45
  const hipY = centerY + 20
  const leftShoulderX = centerX - 25 + sway * 0.2
  const rightShoulderX = centerX + 25 + sway * 0.2
  const leftElbowX = centerX - 47 - punch * 0.35
  const rightElbowX = centerX + 42 + punch
  const wristY = centerY - 12 + Math.cos(pace) * 8

  const landmarks = landmarksFor(index)
  if (landmarks) {
    const points = posePoints(index, landmarks)
    const point = (landmarkIndex: number): [number, number] | undefined => {
      const landmark = points[landmarkIndex]
      if (!landmark || landmark.visibility < .02) return undefined
      return [landmark.x * bounds.width, landmark.y * bounds.height]
    }
    const connections: ReadonlyArray<readonly [number, number]> = index === 0
      ? [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]]
      : [[0, 5], [0, 6], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]
    context.strokeStyle = accent
    for (const [fromIndex, toIndex] of connections) {
      const from = point(fromIndex)
      const to = point(toIndex)
      if (!from || !to) continue
      context.beginPath()
      context.moveTo(...from)
      context.lineTo(...to)
      context.stroke()
    }
    context.fillStyle = accent
    for (const landmarkIndex of new Set(connections.flat())) {
      const landmark = point(landmarkIndex)
      if (!landmark) continue
      context.beginPath()
      context.arc(...landmark, view === 'avatar' ? 5 : 3, 0, Math.PI * 2)
      context.fill()
    }
    context.shadowBlur = 0
    return
  }

  if (view === 'avatar') {
    context.globalAlpha = 0.18
    context.beginPath()
    context.ellipse(centerX + sway * 0.2, centerY - 10, 37, 72, 0, 0, Math.PI * 2)
    context.fill()
    context.globalAlpha = 1
  }
  const lines = [
    [centerX, headY + 15, centerX + sway * 0.2, shoulderY],
    [leftShoulderX, shoulderY, leftElbowX, centerY - 25],
    [leftElbowX, centerY - 25, leftElbowX - 8, wristY],
    [rightShoulderX, shoulderY, rightElbowX, centerY - 25],
    [rightElbowX, centerY - 25, rightElbowX + 6, wristY - 7],
    [leftShoulderX, shoulderY, centerX + sway * 0.2, hipY],
    [rightShoulderX, shoulderY, centerX + sway * 0.2, hipY],
    [centerX + sway * 0.2, hipY, centerX - 25 - sway * 0.3, centerY + 70],
    [centerX + sway * 0.2, hipY, centerX + 25 - sway * 0.3, centerY + 70],
  ]
  for (const [fromX, fromY, toX, toY] of lines) {
    context.beginPath()
    context.moveTo(fromX, fromY)
    context.lineTo(toX, toY)
    context.stroke()
  }
  context.beginPath()
  context.arc(centerX, headY, 15, 0, Math.PI * 2)
  view === 'avatar' ? context.fill() : context.stroke()
  context.shadowBlur = 0
}

function renderPoses(): void {
  const time = sourceVideo.currentTime || 0
  if (comparisonReady) detectMediaPipePose()
  canvasElements.forEach((canvas, index) => drawPlaceholder(canvas, index, time))
  if (view === 'avatar') renderAvatars(time)
  if (!sourceVideo.paused && !sourceVideo.ended) requestAnimationFrame(renderPoses)
}

videoInput.addEventListener('change', () => {
  const file = videoInput.files?.[0]
  if (!file) return
  loadVideoSource(URL.createObjectURL(file), file.name, file)
  runStatus.textContent = 'carregando MediaPipe'
  void loadMediaPipe().then(() => {
    runStatus.textContent = 'video carregado'
    const mediaPipeState = document.querySelector<HTMLElement>('.model-card:first-child .model-state')
    if (mediaPipeState) mediaPipeState.textContent = 'rastreador local pronto'
  }).catch((error: unknown) => {
    console.error('Unable to load MediaPipe pose landmarker', error)
    runStatus.textContent = 'MediaPipe indisponivel'
    const mediaPipeState = document.querySelector<HTMLElement>('.model-card:first-child .model-state')
    if (mediaPipeState) mediaPipeState.textContent = 'falha ao carregar modelo'
  })
})

recordButton.addEventListener('click', () => { void startRecording() })
stopRecordingButton.addEventListener('click', () => { void stopRecording() })
processButton.addEventListener('click', () => { void processWithWebGpuOnly() })

sourceVideo.addEventListener('loadedmetadata', () => {
  timeline.max = String(effectiveDuration())
  timeline.disabled = false
  playButton.disabled = false
  videoEmpty.hidden = true
  updateTimeline()
  renderPoses()
})

sourceVideo.addEventListener('loadeddata', () => {
  processButton.disabled = false
  runStatus.textContent = 'video pronto para comparar'
})

sourceVideo.addEventListener('error', () => {
  processButton.disabled = true
  playButton.disabled = true
  timeline.disabled = true
  runStatus.textContent = 'gravacao nao pode ser aberta'
  processingDetail.textContent = 'O navegador nao conseguiu decodificar esta gravacao. Grave novamente.'
})

sourceVideo.addEventListener('timeupdate', updateTimeline)
sourceVideo.addEventListener('play', () => {
  playButton.textContent = '❚❚'
  runStatus.textContent = 'reproduzindo comparacao'
  renderPoses()
})
sourceVideo.addEventListener('pause', () => {
  playButton.textContent = '▶'
  if (!sourceVideo.ended) runStatus.textContent = 'comparacao pausada'
})
sourceVideo.addEventListener('ended', () => { runStatus.textContent = 'comparacao concluida' })

playButton.addEventListener('click', () => {
  if (sourceVideo.paused) void sourceVideo.play()
  else sourceVideo.pause()
})
timeline.addEventListener('input', () => {
  sourceVideo.currentTime = Number(timeline.value)
  updateTimeline()
  renderPoses()
})

document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((button) => {
  button.addEventListener('click', () => {
    view = button.dataset.view === 'skeleton' ? 'skeleton' : 'avatar'
    labShell.classList.toggle('is-skeleton', view === 'skeleton')
    document.querySelectorAll('[data-view]').forEach((control) => control.classList.toggle('is-active', control === button))
    renderPoses()
  })
})

window.addEventListener('resize', renderPoses)
window.addEventListener('beforeunload', clearCameraStream)
avatarCanvases.forEach((canvas) => avatarPreviews.push(createAvatarPreview(canvas)))
void loadAvatars().then(renderPoses).catch((error: unknown) => {
  console.error('Unable to load Y Bot previews', error)
  runStatus.textContent = 'avatar 3D indisponivel'
})
void checkBackend()
renderPoses()