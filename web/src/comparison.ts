import './comparison.css'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'
import { clone } from 'three/addons/utils/SkeletonUtils.js'
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
} from './pose-retarget'
import { getSampleTimes, processWithWebGpu, SAMPLE_FPS, seekVideo, type WebPoseFrame } from './webgpu-pose'

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
let recorder: MediaRecorder | undefined
let recordingChunks: Blob[] = []
let sourceBlob: Blob | undefined
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
  timeline.value = String(sourceVideo.currentTime)
  timeOutput.value = `${formatTime(sourceVideo.currentTime)} / ${formatTime(sourceVideo.duration)}`
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

function loadVideoSource(url: string, label: string, blob: Blob): void {
  clearCameraStream()
  if (videoUrl) URL.revokeObjectURL(videoUrl)
  videoUrl = url
  labFrames = {}
  comparisonReady = false
  processingStatus.hidden = true
  sourceBlob = blob
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
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    runStatus.textContent = 'gravacao indisponivel neste navegador'
    return
  }
  try {
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
    recorder = new MediaRecorder(
      cameraStream,
      MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? { mimeType: 'video/webm;codecs=vp9' } : undefined,
    )
    recordingChunks = []
    recorder.addEventListener('dataavailable', (event) => { if (event.data.size) recordingChunks.push(event.data) })
    recorder.addEventListener('stop', () => {
      const recording = new Blob(recordingChunks, { type: recorder?.mimeType || 'video/webm' })
      loadVideoSource(
        URL.createObjectURL(recording),
        `gravacao-${new Date().toLocaleTimeString('pt-BR').replaceAll(':', '-')}.webm`,
        recording,
      )
      runStatus.textContent = 'gravacao pronta para comparar'
      recordButton.disabled = false
      stopRecordingButton.disabled = true
    })
    recorder.start(250)
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
    })

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

    document.querySelectorAll<HTMLElement>('.model-card').forEach((card, index) => {
      const frames = labFrames[modelDefinitions[index].pipeline]
      const trackedCount = frames?.filter((frame) => frame.tracked).length ?? 0
      const state = card.querySelector<HTMLElement>('.model-state')
      const frameRate = card.querySelector<HTMLElement>('.frame-rate')
      if (state) state.textContent = trackedCount ? `${trackedCount} poses ok` : 'sem pose util'
      if (frameRate) frameRate.textContent = `${SAMPLE_FPS} FPS`
      card.classList.remove('is-processing')
    })

    updateProcessingStatus({
      status: 'complete',
      stage: 'complete',
      progress: 100,
      detail: 'Comparacao pronta — hybrid / filtro / crop lado a lado com o mesmo IK',
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
stopRecordingButton.addEventListener('click', () => recorder?.state === 'recording' && recorder.stop())
processButton.addEventListener('click', () => { void processComparison() })

sourceVideo.addEventListener('loadedmetadata', () => {
  timeline.max = String(sourceVideo.duration)
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
