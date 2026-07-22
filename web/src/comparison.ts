import './comparison.css'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { clone } from 'three/addons/utils/SkeletonUtils.js'
import { ArrayBufferTarget, Muxer } from 'mp4-muxer'
import {
  connectionsFor,
  projectH36mToSticker,
  trackedFromBlazePoseWorld,
  trackedFromCoco2d,
  trackedFromH36m3d,
  trackedFromHybrid,
  type LabPoseFrame,
  type StickerLandmark,
  type Topology,
} from './pose-adapters'
import { smoothLabFrames } from './pose-filter'
import {
  applyGuardPose,
  applyUpperBodyPose,
  prepareAvatarRig,
  type AvatarRig,
  type TrackedPose,
} from './pose-retarget'
import { getSampleTimes, processWithWebGpu, SAMPLE_FPS, seekVideo, type WebPoseFrame } from './webgpu-pose'

declare const MediaStreamTrackProcessor: new (init: { track: MediaStreamTrack }) => { readable: ReadableStream<VideoFrame> }

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <main class="lab-shell">
    <header class="lab-header">
      <a class="brand" href="/" aria-label="Voltar ao Counterpunch">COUNTERPUNCH <span>LAB</span></a>
      <p>BAKE-OFF PARA O JOGO</p>
      <div class="run-status"><i></i><span id="run-status">aguardando video</span></div>
    </header>

    <section class="source-zone" aria-labelledby="source-title">
      <div class="source-copy">
        <p class="section-kicker">FONTE</p>
        <h1 id="source-title">Seis experimentos. Mesmo video.</h1>
        <p>Baseline MediaPipe vs RTMPose, lift puro, hybrid XY-lock, filtro temporal e crop top-down — mesmo sampling e mesmo IK Mixamo.</p>
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
          <p id="processing-detail">MediaPipe + RTMPose + hybrid/crop no mesmo sampling.</p>
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
      <div class="model-count"><strong>06</strong><span>candidatos</span></div>
      <div class="view-toggle" role="group" aria-label="Visualizacao dos resultados">
        <button type="button" class="is-active" data-view="avatar">MODELO 3D</button>
        <button type="button" data-view="skeleton">STICKER</button>
      </div>
      <p id="pipeline-note" class="pipeline-note">Verificando WebGPU...</p>
    </section>

    <section id="model-grid" class="model-grid" aria-label="Resultados dos modelos"></section>
  </main>
`

const labShell = document.querySelector<HTMLElement>('.lab-shell')!

const modelDefinitions = [
  {
    name: 'MediaPipe',
    score: 'live Z',
    color: '#5cd5bf',
    pipeline: 'mediapipe',
    blurb: 'world landmarks nativos',
  },
  {
    name: 'RTMPose',
    score: '2D+Z*',
    color: '#e6bd68',
    pipeline: 'rtmpose',
    blurb: 'heuristica mildDepth',
  },
  {
    name: 'AGFormer puro',
    score: 'lift 3D',
    color: '#7da9f5',
    pipeline: 'agformer_pure',
    blurb: 'H36M raw → IK',
  },
  {
    name: 'Hybrid XY+Z',
    score: 'XY lock',
    color: '#d67ad4',
    pipeline: 'hybrid',
    blurb: 'RTM XY + AGFormer Z',
  },
  {
    name: 'Hybrid + filtro',
    score: 'OneEuro',
    color: '#f0a06a',
    pipeline: 'hybrid_smooth',
    blurb: 'hybrid + bone lengths',
  },
  {
    name: 'Crop + Hybrid',
    score: 'top-down',
    color: '#8fd18a',
    pipeline: 'crop_hybrid',
    blurb: 'bbox → RTM → hybrid',
  },
] as const

type PipelineKey = (typeof modelDefinitions)[number]['pipeline']

const modelGrid = document.querySelector<HTMLDivElement>('#model-grid')!
modelGrid.innerHTML = modelDefinitions.map((model, index) => `
  <article class="model-card" style="--accent: ${model.color}; --delay: ${index * 0.08}s">
    <header>
      <div><p>${model.blurb}</p><h2>${model.name}</h2></div>
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
let cameraStream: MediaStream | undefined
type CameraRecorder = { stop: () => Promise<Blob> }
let cameraRecorder: CameraRecorder | undefined
let sourceBlob: Blob | undefined
let sourceDuration: number | undefined
let recordingStartedAt = 0
let labFrames: Partial<Record<PipelineKey, LabPoseFrame[]>> = {}
let comparisonReady = false

type ProcessJob = {
  status: 'processing' | 'complete' | 'failed'
  stage: string
  progress: number
  detail: string
}

type AvatarPreview = {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  rig?: AvatarRig
}

const avatarPreviews: AvatarPreview[] = []
const avatarLoader = new FBXLoader()
let avatarTemplate: THREE.Group | undefined

function formatTime(seconds: number): string {
  const wholeSeconds = Number.isFinite(seconds) ? Math.floor(seconds) : 0
  return `${String(Math.floor(wholeSeconds / 60)).padStart(2, '0')}:${String(wholeSeconds % 60).padStart(2, '0')}`
}

function updateTimeline(): void {
  const duration = Number.isFinite(sourceVideo.duration) && sourceVideo.duration > 0
    ? sourceVideo.duration
    : sourceDuration ?? 0
  timeline.value = String(sourceVideo.currentTime)
  timeOutput.value = `${formatTime(sourceVideo.currentTime)} / ${formatTime(duration)}`
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

async function loadAvatars(): Promise<void> {
  if (avatarTemplate) return
  avatarTemplate = await avatarLoader.loadAsync('/assets/mixamo/character/y-bot.fbx')
  avatarPreviews.forEach((preview) => {
    const avatar = clone(avatarTemplate!) as THREE.Group
    avatar.position.y = 0
    avatar.rotation.y = Math.PI
    preview.scene.add(avatar)
    preview.rig = prepareAvatarRig(avatar)
  })
}

function nearestFrame(frames: LabPoseFrame[] | undefined, time: number): LabPoseFrame | undefined {
  if (!frames?.length) return undefined
  return frames.reduce((current, frame) =>
    Math.abs(frame.time - time) < Math.abs(current.time - time) ? frame : current)
}

function frameFor(index: number): LabPoseFrame | undefined {
  return nearestFrame(labFrames[modelDefinitions[index].pipeline], sourceVideo.currentTime)
}

function poseAvatar(rig: AvatarRig, index: number): void {
  const frame = frameFor(index)
  if (frame?.tracked) applyUpperBodyPose(rig, frame.tracked)
  else applyGuardPose(rig)
}

function renderAvatars(): void {
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
    if (preview.rig) poseAvatar(preview.rig, avatarPreviews.indexOf(preview))
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
  labFrames = {}
  comparisonReady = false
  processingStatus.hidden = true
  sourceBlob = blob
  sourceDuration = duration
  sourceVideo.srcObject = null
  sourceVideo.src = url
  sourceVideo.load()
  fileName.textContent = label
  document.querySelectorAll<HTMLElement>('.model-state').forEach((state) => { state.textContent = 'video carregado — processe' })
  document.querySelectorAll<HTMLElement>('.frame-rate').forEach((rate) => { rate.textContent = '-- FPS' })
}

function updateProcessingStatus(job: ProcessJob): void {
  processingStatus.hidden = false
  processingStage.textContent = job.stage === 'queued' ? 'Na fila' : job.stage
  processingPercent.textContent = `${job.progress}%`
  processingProgress.value = job.progress
  processingDetail.textContent = job.detail
  pipelineNote.textContent = job.detail
  document.querySelectorAll<HTMLElement>('.model-card').forEach((card, index) => {
    const pipeline = modelDefinitions[index].pipeline
    const detail = job.detail.toLowerCase()
    const isActive = job.stage === pipeline
      || (pipeline === 'mediapipe' && detail.includes('mediapipe'))
      || (pipeline === 'rtmpose' && detail.includes('rtmpose full-frame'))
      || (pipeline === 'agformer_pure' && detail.includes('agformer') && detail.includes('full-frame'))
      || (pipeline === 'hybrid' && detail.includes('agformer') && detail.includes('full-frame'))
      || (pipeline === 'hybrid_smooth' && detail.includes('concluidos'))
      || (pipeline === 'crop_hybrid' && detail.includes('crop'))
    card.classList.toggle('is-processing', Boolean(isActive) && job.status === 'processing')
    if (isActive && job.status === 'processing') card.querySelector<HTMLElement>('.model-state')!.textContent = 'processando agora'
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
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    })
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
    loadVideoSource(
      URL.createObjectURL(recording),
      `gravacao-${new Date().toLocaleTimeString('pt-BR').replaceAll(':', '-')}.mp4`,
      recording,
      duration,
    )
    runStatus.textContent = 'gravacao pronta para comparar'
    recordButton.disabled = false
  } catch (error) {
    console.error('Unable to finalize camera recording', error)
    runStatus.textContent = 'falha ao finalizar gravacao'
    recordButton.disabled = false
    stopRecordingButton.disabled = true
  }
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

async function sampleMediaPipe(video: HTMLVideoElement, onProgress: (detail: string, progress: number) => void): Promise<LabPoseFrame[]> {
  await loadMediaPipe()
  if (!poseLandmarker) throw new Error('MediaPipe indisponivel')
  if (!Number.isFinite(video.duration) || video.duration <= 0) {
    throw new Error('Video sem duracao finita para amostrar o MediaPipe.')
  }
  const times = getSampleTimes(video.duration)
  const originalTime = video.currentTime
  const frames: LabPoseFrame[] = []
  let timestamp = 1
  for (let index = 0; index < times.length; index += 1) {
    await seekVideo(video, times[index])
    timestamp += 1000 / SAMPLE_FPS
    const result = poseLandmarker.detectForVideo(video, timestamp)
    const image = result.landmarks[0] as StickerLandmark[] | undefined
    const world = result.worldLandmarks[0]
    frames.push({
      time: video.currentTime,
      sticker: image ? { topology: 'blazepose', landmarks: image } : null,
      tracked: world && image ? trackedFromBlazePoseWorld(world, image) : world ? trackedFromBlazePoseWorld(world) : null,
    })
    onProgress(`MediaPipe: amostra ${index + 1} de ${times.length}`, Math.round(index / times.length * 100))
  }
  await seekVideo(video, originalTime)
  return frames
}

function labFramesFromRtmpose(frames: WebPoseFrame[]): LabPoseFrame[] {
  return frames.map((frame) => ({
    time: frame.time,
    sticker: frame.landmarks ? { topology: 'coco' as Topology, landmarks: frame.landmarks } : null,
    tracked: frame.landmarks ? trackedFromCoco2d(frame.landmarks) : null,
  }))
}

function labFramesFromAgformer(frames: WebPoseFrame[]): LabPoseFrame[] {
  return frames.map((frame) => {
    if (!frame.landmarks) return { time: frame.time, sticker: null, tracked: null }
    return {
      time: frame.time,
      sticker: { topology: 'h36m' as Topology, landmarks: projectH36mToSticker(frame.landmarks) },
      tracked: trackedFromH36m3d(frame.landmarks),
    }
  })
}

function labFramesFromHybrid(cocoFrames: WebPoseFrame[], h36mFrames: WebPoseFrame[]): LabPoseFrame[] {
  return cocoFrames.map((frame, index) => {
    const h36m = h36mFrames[index]?.landmarks
    if (!frame.landmarks || !h36m) return { time: frame.time, sticker: null, tracked: null }
    return {
      time: frame.time,
      sticker: { topology: 'coco' as Topology, landmarks: frame.landmarks },
      tracked: trackedFromHybrid(frame.landmarks, h36m),
    }
  })
}

/** Wrist Z relative to shoulder midplane. Negative = in front (toward camera), matching MediaPipe. */
function wristDepth(pose: TrackedPose): { left: number; right: number; mean: number } {
  const shoulderZ = (pose.leftShoulder.z + pose.rightShoulder.z) / 2
  const left = pose.leftWrist.z - shoulderZ
  const right = pose.rightWrist.z - shoulderZ
  return { left, right, mean: (left + right) / 2 }
}

/** Wrist Y relative to shoulder midplane. Positive = above shoulders (MediaPipe +Y up). */
function wristLift(pose: TrackedPose): number {
  const shoulderY = (pose.leftShoulder.y + pose.rightShoulder.y) / 2
  return ((pose.leftWrist.y + pose.rightWrist.y) / 2) - shoulderY
}

type DepthDiag = {
  pipeline: PipelineKey
  frames: number
  /** Candidate wrist in back (+Z) while MediaPipe has wrists in front (−Z). */
  frontBackFlips: number
  /** Mean |Δ wrist-depth| vs MediaPipe (meters). */
  meanAbsDepthDelta: number
  /** Mean wrist lift delta vs MediaPipe (meters); large positive = hands too high. */
  meanLiftDelta: number
  sampleConflicts: Array<{ time: number; mpDepth: number; candDepth: number; liftDelta: number }>
}

function diagnoseDepthVsMediaPipe(
  mediaPipe: LabPoseFrame[],
  candidate: LabPoseFrame[],
  pipeline: PipelineKey,
): DepthDiag {
  let frames = 0
  let frontBackFlips = 0
  let absDepthSum = 0
  let liftDeltaSum = 0
  const sampleConflicts: DepthDiag['sampleConflicts'] = []

  const count = Math.min(mediaPipe.length, candidate.length)
  for (let index = 0; index < count; index += 1) {
    const mp = mediaPipe[index]?.tracked
    const cand = candidate[index]?.tracked
    if (!mp || !cand) continue
    frames += 1
    const mpDepth = wristDepth(mp).mean
    const candDepth = wristDepth(cand).mean
    const liftDelta = wristLift(cand) - wristLift(mp)
    absDepthSum += Math.abs(candDepth - mpDepth)
    liftDeltaSum += liftDelta
    // MediaPipe clearly in front, candidate clearly behind (sign disagreement with margin).
    if (mpDepth < -0.02 && candDepth > 0.02) {
      frontBackFlips += 1
      if (sampleConflicts.length < 5) {
        sampleConflicts.push({
          time: mediaPipe[index].time,
          mpDepth,
          candDepth,
          liftDelta,
        })
      }
    }
  }

  return {
    pipeline,
    frames,
    frontBackFlips,
    meanAbsDepthDelta: frames ? absDepthSum / frames : 0,
    meanLiftDelta: frames ? liftDeltaSum / frames : 0,
    sampleConflicts,
  }
}

function logDepthDiagnostics(mediaPipe: LabPoseFrame[]): void {
  const keys = modelDefinitions.map((model) => model.pipeline).filter((key) => key !== 'mediapipe')
  const reports = keys
    .map((key) => {
      const frames = labFrames[key]
      return frames ? diagnoseDepthVsMediaPipe(mediaPipe, frames, key) : null
    })
    .filter((report): report is DepthDiag => Boolean(report))

  console.group('[lab] depth vs MediaPipe (−Z = frente / +Z = costas)')
  for (const report of reports) {
    const flipPct = report.frames ? Math.round((report.frontBackFlips / report.frames) * 100) : 0
    console.log(
      `${report.pipeline}: flips frente→costas ${report.frontBackFlips}/${report.frames} (${flipPct}%)`,
      `|ΔZ|≈${report.meanAbsDepthDelta.toFixed(3)}m`,
      `liftΔ≈${report.meanLiftDelta.toFixed(3)}m`,
      report.sampleConflicts,
    )
  }
  console.groupEnd()

  const worst = [...reports].sort((a, b) => b.frontBackFlips - a.frontBackFlips)[0]
  if (worst && worst.frontBackFlips > 0) {
    const flipPct = Math.round((worst.frontBackFlips / Math.max(1, worst.frames)) * 100)
    pipelineNote.textContent =
      `Debug Z: ${worst.pipeline} inverte frente/costas em ${flipPct}% dos frames` +
      (worst.meanLiftDelta > 0.05 ? ` · mãos +${worst.meanLiftDelta.toFixed(2)}m vs MP` : '')
  } else if (reports.length) {
    const lift = [...reports].sort((a, b) => Math.abs(b.meanLiftDelta) - Math.abs(a.meanLiftDelta))[0]
    pipelineNote.textContent = lift && Math.abs(lift.meanLiftDelta) > 0.04
      ? `Debug Z ok (sem flip). Lift suspeito: ${lift.pipeline} ΔY≈${lift.meanLiftDelta.toFixed(2)}m`
      : 'Debug Z: sem flip frente/costas vs MediaPipe'
  }
}

type JointSnapshot = {
  leftShoulder: { x: number; y: number; z: number }
  rightShoulder: { x: number; y: number; z: number }
  leftElbow: { x: number; y: number; z: number }
  rightElbow: { x: number; y: number; z: number }
  leftWrist: { x: number; y: number; z: number }
  rightWrist: { x: number; y: number; z: number }
  wristDepth: number
  wristLift: number
  elbowDepth: number
  /** Same axes the Mixamo IK consumes (poseToAvatarSpace). */
  avatarLeftWrist: { x: number; y: number; z: number }
  avatarRightWrist: { x: number; y: number; z: number }
  avatarLeftElbow: { x: number; y: number; z: number }
  avatarRightElbow: { x: number; y: number; z: number }
}

function toAvatarSpace(
  point: { x: number; y: number; z: number },
  shoulderCenter: { x: number; y: number; z: number },
  shoulderWidth: number,
): { x: number; y: number; z: number } {
  return {
    x: Number(((point.x - shoulderCenter.x) / shoulderWidth).toFixed(4)),
    y: Number(((shoulderCenter.y - point.y) / shoulderWidth).toFixed(4)),
    z: Number(((shoulderCenter.z - point.z) / shoulderWidth).toFixed(4)),
  }
}

function snapshotPose(pose: TrackedPose): JointSnapshot {
  const shoulderCenter = {
    x: (pose.leftShoulder.x + pose.rightShoulder.x) / 2,
    y: (pose.leftShoulder.y + pose.rightShoulder.y) / 2,
    z: (pose.leftShoulder.z + pose.rightShoulder.z) / 2,
  }
  const round = (point: { x: number; y: number; z: number }) => ({
    x: Number(point.x.toFixed(4)),
    y: Number(point.y.toFixed(4)),
    z: Number(point.z.toFixed(4)),
  })
  return {
    leftShoulder: round(pose.leftShoulder),
    rightShoulder: round(pose.rightShoulder),
    leftElbow: round(pose.leftElbow),
    rightElbow: round(pose.rightElbow),
    leftWrist: round(pose.leftWrist),
    rightWrist: round(pose.rightWrist),
    wristDepth: Number((((pose.leftWrist.z + pose.rightWrist.z) / 2) - shoulderCenter.z).toFixed(4)),
    wristLift: Number((((pose.leftWrist.y + pose.rightWrist.y) / 2) - shoulderCenter.y).toFixed(4)),
    elbowDepth: Number((((pose.leftElbow.z + pose.rightElbow.z) / 2) - shoulderCenter.z).toFixed(4)),
    avatarLeftWrist: toAvatarSpace(pose.leftWrist, shoulderCenter, pose.shoulderWidth),
    avatarRightWrist: toAvatarSpace(pose.rightWrist, shoulderCenter, pose.shoulderWidth),
    avatarLeftElbow: toAvatarSpace(pose.leftElbow, shoulderCenter, pose.shoulderWidth),
    avatarRightElbow: toAvatarSpace(pose.rightElbow, shoulderCenter, pose.shoulderWidth),
  }
}

/** Full audit payload for Playwright / console — compare TrackedPose joints vs MediaPipe. */
function buildLabAudit() {
  const mediaPipe = labFrames.mediapipe ?? []
  const sampleIndexes = mediaPipe.length
    ? [0, Math.floor(mediaPipe.length * 0.25), Math.floor(mediaPipe.length * 0.5), Math.floor(mediaPipe.length * 0.75), mediaPipe.length - 1]
      .filter((value, index, all) => all.indexOf(value) === index)
    : []

  const pipelines = modelDefinitions.map((model) => model.pipeline)
  const summaries = pipelines.map((pipeline) => {
    const frames = labFrames[pipeline] ?? []
    const vsMp = pipeline === 'mediapipe' || !mediaPipe.length
      ? null
      : diagnoseDepthVsMediaPipe(mediaPipe, frames, pipeline)
    let meanWristDepth = 0
    let meanWristLift = 0
    let tracked = 0
    let handsFront = 0
    let handsBack = 0
    let handsHigh = 0
    for (const frame of frames) {
      if (!frame.tracked) continue
      tracked += 1
      const depth = wristDepth(frame.tracked).mean
      const lift = wristLift(frame.tracked)
      meanWristDepth += depth
      meanWristLift += lift
      if (depth < -0.02) handsFront += 1
      if (depth > 0.02) handsBack += 1
      if (lift > -0.05) handsHigh += 1 // near/above shoulders (fists at chest should be clearly below)
    }
    return {
      pipeline,
      tracked,
      meanWristDepth: tracked ? meanWristDepth / tracked : null,
      meanWristLift: tracked ? meanWristLift / tracked : null,
      handsFront,
      handsBack,
      handsHigh,
      vsMediaPipe: vsMp,
    }
  })

  const samples = sampleIndexes.map((index) => {
    const time = mediaPipe[index]?.time ?? labFrames.rtmpose?.[index]?.time ?? index / SAMPLE_FPS
    const byPipeline: Record<string, JointSnapshot | null> = {}
    for (const pipeline of pipelines) {
      const pose = labFrames[pipeline]?.[index]?.tracked ?? null
      byPipeline[pipeline] = pose ? snapshotPose(pose) : null
    }
    return { index, time, byPipeline }
  })

  return {
    ready: comparisonReady,
    frameCount: mediaPipe.length,
    sampleFps: SAMPLE_FPS,
    convention: 'TrackedPose meters: +Y up, −Z toward camera / in front, +Z behind',
    summaries,
    samples,
  }
}

declare global {
  interface Window {
    __labAudit?: () => ReturnType<typeof buildLabAudit>
    __labFrames?: typeof labFrames
  }
}
window.__labAudit = buildLabAudit
Object.defineProperty(window, '__labFrames', {
  configurable: true,
  get: () => labFrames,
})

async function processComparison(): Promise<void> {
  if (!sourceBlob) return
  processButton.disabled = true
  comparisonReady = false
  runStatus.textContent = 'processando candidatos'
  updateProcessingStatus({ status: 'processing', stage: 'mediapipe', progress: 0, detail: 'Amostrando MediaPipe no mesmo clock do WebGPU' })
  try {
    const mediaPipeFrames = await sampleMediaPipe(sourceVideo, (detail, progress) => {
      updateProcessingStatus({ status: 'processing', stage: 'mediapipe', progress: Math.round(progress * 0.22), detail })
    })

    const webGpu = await processWithWebGpu(sourceVideo, (detail, progress) => {
      const lower = detail.toLowerCase()
      const stage = lower.includes('crop')
        ? 'crop_hybrid'
        : lower.includes('agformer')
          ? 'agformer_pure'
          : 'rtmpose'
      updateProcessingStatus({
        status: 'processing',
        stage,
        progress: 22 + Math.round(progress * 0.76),
        detail,
      })
    }, sourceDuration)

    const hybrid = labFramesFromHybrid(webGpu.rtmpose, webGpu.motionagformer)
    labFrames = {
      mediapipe: mediaPipeFrames,
      rtmpose: labFramesFromRtmpose(webGpu.rtmpose),
      agformer_pure: labFramesFromAgformer(webGpu.motionagformer),
      hybrid,
      hybrid_smooth: smoothLabFrames(hybrid),
      crop_hybrid: labFramesFromHybrid(webGpu.rtmposeCrop, webGpu.agformerCrop),
    }
    comparisonReady = true
    logDepthDiagnostics(mediaPipeFrames)
    console.log('[lab] audit', buildLabAudit())

    document.querySelectorAll<HTMLElement>('.model-card').forEach((card, index) => {
      const pipeline = modelDefinitions[index].pipeline
      const frames = labFrames[pipeline]
      const trackedCount = frames?.filter((frame) => frame.tracked).length ?? 0
      const state = card.querySelector<HTMLElement>('.model-state')
      const frameRate = card.querySelector<HTMLElement>('.frame-rate')
      if (pipeline !== 'mediapipe' && frames) {
        const diag = diagnoseDepthVsMediaPipe(mediaPipeFrames, frames, pipeline)
        const flipPct = diag.frames ? Math.round((diag.frontBackFlips / diag.frames) * 100) : 0
        if (state) {
          state.textContent = diag.frontBackFlips
            ? `${trackedCount} ok · Z flip ${flipPct}%`
            : trackedCount
              ? `${trackedCount} poses ok`
              : 'sem pose util'
        }
      } else if (state) {
        state.textContent = trackedCount ? `${trackedCount} poses ok` : 'sem pose util'
      }
      if (frameRate) frameRate.textContent = `${SAMPLE_FPS} FPS`
      card.classList.remove('is-processing')
    })

    updateProcessingStatus({
      status: 'complete',
      stage: 'complete',
      progress: 100,
      detail: 'Comparacao pronta — abra o console para o log de profundidade vs MediaPipe',
    })
    runStatus.textContent = 'comparacao pronta'
    renderPoses()
  } catch (error) {
    console.error('Unable to process comparison', error)
    runStatus.textContent = 'falha no processamento'
    processingDetail.textContent = error instanceof Error ? error.message : 'Falha desconhecida durante o processamento.'
  } finally {
    processButton.disabled = false
  }
}

async function checkBackend(): Promise<void> {
  const webGpuAvailable = 'gpu' in navigator && Boolean(await navigator.gpu?.requestAdapter())
  pipelineNote.textContent = webGpuAvailable
    ? 'WebGPU ativo. Candidatos: MediaPipe, RTMPose, AGFormer, Hybrid, Filtro, Crop.'
    : 'WebGPU indisponivel: RTMPose/AGFormer usam fallback WASM.'
}

function drawSticker(canvas: HTMLCanvasElement, index: number): void {
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
  context.strokeStyle = accent
  context.fillStyle = accent
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.lineWidth = 3
  context.shadowColor = `${accent}55`
  context.shadowBlur = 14

  if (!comparisonReady) {
    context.globalAlpha = 0.55
    context.font = '11px ui-monospace, monospace'
    context.textAlign = 'center'
    context.fillText('PROCESSE O VIDEO', bounds.width / 2, bounds.height / 2)
    context.shadowBlur = 0
    context.globalAlpha = 1
    return
  }

  const frame = frameFor(index)
  const sticker = frame?.sticker
  if (!sticker) {
    context.globalAlpha = 0.45
    context.font = '11px ui-monospace, monospace'
    context.textAlign = 'center'
    context.fillText('SEM DETECCAO', bounds.width / 2, bounds.height / 2)
    context.shadowBlur = 0
    context.globalAlpha = 1
    return
  }

  const point = (landmarkIndex: number): [number, number] | undefined => {
    const landmark = sticker.landmarks[landmarkIndex]
    if (!landmark || (landmark.visibility ?? 1) < 0.12) return undefined
    if (!Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) return undefined
    if (landmark.x < -0.25 || landmark.x > 1.25 || landmark.y < -0.25 || landmark.y > 1.25) return undefined
    return [landmark.x * bounds.width, landmark.y * bounds.height]
  }

  const connections = connectionsFor(sticker.topology)
  for (const [fromIndex, toIndex] of connections) {
    const from = point(fromIndex)
    const to = point(toIndex)
    if (!from || !to) continue
    context.beginPath()
    context.moveTo(...from)
    context.lineTo(...to)
    context.stroke()
  }
  for (const landmarkIndex of new Set(connections.flat())) {
    const landmark = point(landmarkIndex)
    if (!landmark) continue
    context.beginPath()
    context.arc(...landmark, 3.2, 0, Math.PI * 2)
    context.fill()
  }
  context.shadowBlur = 0
}

function renderPoses(): void {
  canvasElements.forEach((canvas, index) => drawSticker(canvas, index))
  if (view === 'avatar') renderAvatars()
  if (!sourceVideo.paused && !sourceVideo.ended) requestAnimationFrame(renderPoses)
}

videoInput.addEventListener('change', () => {
  const file = videoInput.files?.[0]
  if (!file) return
  loadVideoSource(URL.createObjectURL(file), file.name, file)
  runStatus.textContent = 'carregando MediaPipe'
  void loadMediaPipe().then(() => {
    runStatus.textContent = 'video carregado — processe a comparacao'
  }).catch((error: unknown) => {
    console.error('Unable to load MediaPipe pose landmarker', error)
    runStatus.textContent = 'MediaPipe indisponivel'
  })
})

recordButton.addEventListener('click', () => { void startRecording() })
stopRecordingButton.addEventListener('click', () => { void stopRecording() })
processButton.addEventListener('click', () => { void processComparison() })

sourceVideo.addEventListener('loadedmetadata', () => {
  const duration = Number.isFinite(sourceVideo.duration) && sourceVideo.duration > 0
    ? sourceVideo.duration
    : sourceDuration ?? 0
  timeline.max = String(Math.max(duration, 0.001))
  timeline.disabled = false
  playButton.disabled = false
  processButton.disabled = false
  videoEmpty.hidden = true
  updateTimeline()
  renderPoses()
})

sourceVideo.addEventListener('timeupdate', updateTimeline)
sourceVideo.addEventListener('play', () => {
  playButton.textContent = '❚❚'
  runStatus.textContent = comparisonReady ? 'reproduzindo comparacao' : 'reproduzindo video'
  renderPoses()
})
sourceVideo.addEventListener('pause', () => {
  playButton.textContent = '▶'
  if (!sourceVideo.ended) runStatus.textContent = comparisonReady ? 'comparacao pausada' : 'video pausado'
})
sourceVideo.addEventListener('ended', () => { runStatus.textContent = 'reproducao concluida' })

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
