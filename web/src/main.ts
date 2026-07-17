import './style.css'
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import * as THREE from 'three'
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js'

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <main id="game-shell">
    <canvas id="ring" aria-label="Protótipo 3D de ringue de boxe"></canvas>
    <header class="hud top-hud">
      <p class="eyebrow">ROUND 01 <span>•</span> PROTOTYPE</p>
      <h1>COUNTERPUNCH</h1>
    </header>
    <section class="hud fighter-status player-status" aria-label="Status do jogador">
      <div><span>PLAYER</span><strong id="player-stamina">100</strong></div>
      <div class="meter"><i id="player-meter"></i></div>
    </section>
    <section class="hud fighter-status opponent-status" aria-label="Status do oponente">
      <div><span>SPARRING BOT</span><strong id="opponent-stamina">100</strong></div>
      <div class="meter"><i id="opponent-meter"></i></div>
    </section>
    <section class="camera-panel" aria-label="Controle de webcam">
      <video id="webcam" autoplay muted playsinline></video>
      <div class="camera-panel__status"><i id="tracking-dot"></i><span id="tracking-status">camera desligada</span></div>
      <button id="camera-button" type="button">Ativar camera</button>
    </section>
    <footer class="hud controls">WASD move &nbsp; · &nbsp; J jab &nbsp; · &nbsp; K cross &nbsp; · &nbsp; R reset</footer>
  </main>
`

const canvas = document.querySelector<HTMLCanvasElement>('#ring')!
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.outputColorSpace = THREE.SRGBColorSpace

const scene = new THREE.Scene()
scene.background = new THREE.Color('#15191c')
scene.fog = new THREE.Fog('#15191c', 10, 28)

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100)
camera.position.set(8.2, 6.8, 10.5)
camera.lookAt(0, 0.5, 0)

scene.add(new THREE.HemisphereLight('#b8e8ef', '#27202b', 2.3))
const keyLight = new THREE.SpotLight('#fff2dc', 180, 25, Math.PI / 5, 0.45)
keyLight.position.set(-3, 9, 5)
keyLight.castShadow = true
scene.add(keyLight)

const ring = new THREE.Group()
scene.add(ring)

const canvasFloor = new THREE.Mesh(
  new THREE.BoxGeometry(8.2, 0.25, 8.2),
  new THREE.MeshStandardMaterial({ color: '#d9d0c0', roughness: 0.86 }),
)
canvasFloor.receiveShadow = true
ring.add(canvasFloor)

const apron = new THREE.Mesh(
  new THREE.BoxGeometry(9.3, 0.55, 9.3),
  new THREE.MeshStandardMaterial({ color: '#1e3437', roughness: 0.62 }),
)
apron.position.y = -0.38
apron.receiveShadow = true
ring.add(apron)

const ropeMaterial = new THREE.MeshStandardMaterial({ color: '#b63236', roughness: 0.42 })
for (const height of [0.8, 1.35, 1.9]) {
  for (const [width, rotation] of [[8.8, 0], [8.8, Math.PI / 2]] as const) {
    for (const side of [-1, 1]) {
      const rope = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, width, 10), ropeMaterial)
      rope.rotation.z = rotation
      rope.position.set(rotation ? side * 4.4 : 0, height, rotation ? 0 : side * 4.4)
      ring.add(rope)
    }
  }
}

function createFighter(bodyColor: string, gloveColor: string): THREE.Group {
  const fighter = new THREE.Group()
  const material = new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 0.63 })
  const gloveMaterial = new THREE.MeshStandardMaterial({ color: gloveColor, roughness: 0.35, metalness: 0.05 })
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.46, 1.0, 8, 16), material)
  torso.position.y = 1.2
  torso.castShadow = true
  fighter.add(torso)
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.34, 16, 12), material)
  head.position.y = 2.08
  head.castShadow = true
  fighter.add(head)
  for (const side of [-1, 1]) {
    const glove = new THREE.Mesh(new THREE.SphereGeometry(0.26, 16, 12), gloveMaterial)
    glove.name = side < 0 ? 'left-glove' : 'right-glove'
    glove.position.set(side * 0.54, 1.55, 0.08)
    glove.scale.set(0.85, 1.12, 1.05)
    glove.castShadow = true
    fighter.add(glove)
  }
  return fighter
}

const player = createFighter('#1f8c87', '#e7b96a')
player.position.set(0, 0, 2.2)
scene.add(player)
let playerAvatar: THREE.Group | undefined
let playerMixer: THREE.AnimationMixer | undefined

type PosePoint = { x: number; y: number; z: number }
type TrackedPose = {
  leftShoulder: PosePoint
  rightShoulder: PosePoint
  leftElbow: PosePoint
  rightElbow: PosePoint
  leftWrist: PosePoint
  rightWrist: PosePoint
  leftHip: PosePoint
  rightHip: PosePoint
  shoulderWidth: number
}
type BoneRestPose = {
  bone: THREE.Bone
  restLocalQuaternion: THREE.Quaternion
  restQuaternionInRoot: THREE.Quaternion
  restDirectionInRoot: THREE.Vector3
}

const retargetBones = new Map<string, BoneRestPose>()
let latestPose: TrackedPose | undefined
const cameraGloves = new THREE.Group()
const cameraGloveMaterial = new THREE.MeshStandardMaterial({ color: '#e2b05f', roughness: 0.3, metalness: 0.06 })
const trackedLeftGlove = new THREE.Mesh(new THREE.SphereGeometry(0.27, 20, 16), cameraGloveMaterial)
const trackedRightGlove = trackedLeftGlove.clone()
for (const glove of [trackedLeftGlove, trackedRightGlove]) {
  glove.scale.set(0.9, 1.15, 1.05)
  glove.castShadow = true
  cameraGloves.add(glove)
}
cameraGloves.visible = false
scene.add(cameraGloves)
const opponent = createFighter('#9a3d3c', '#f1e2c8')
opponent.position.set(0, 0, -2.2)
opponent.rotation.y = Math.PI
scene.add(opponent)

const keys = new Set<string>()
const webcam = document.querySelector<HTMLVideoElement>('#webcam')!
const cameraButton = document.querySelector<HTMLButtonElement>('#camera-button')!
const trackingStatus = document.querySelector<HTMLSpanElement>('#tracking-status')!
const trackingDot = document.querySelector<HTMLElement>('#tracking-dot')!
let playerStamina = 100
let opponentStamina = 100
let lastTime = performance.now()
const fbxLoader = new FBXLoader()
let poseLandmarker: PoseLandmarker | undefined
let lastVideoTime = -1
let trackingActive = false
const trackedGloveTargets = {
  left: new THREE.Vector3(-0.54, 1.55, 0.08),
  right: new THREE.Vector3(0.54, 1.55, 0.08),
}

function firstChildBone(bone: THREE.Bone): THREE.Bone | undefined {
  return bone.children.find((child): child is THREE.Bone => child instanceof THREE.Bone)
}

function cacheRetargetBones(avatar: THREE.Group): void {
  const boneNames = ['mixamorigSpine', 'mixamorigLeftArm', 'mixamorigLeftForeArm', 'mixamorigRightArm', 'mixamorigRightForeArm']
  const rootWorldQuaternion = avatar.getWorldQuaternion(new THREE.Quaternion())
  const inverseRootQuaternion = rootWorldQuaternion.clone().invert()

  for (const name of boneNames) {
    const bone = avatar.getObjectByName(name)
    if (!(bone instanceof THREE.Bone)) continue
    const childBone = firstChildBone(bone)
    if (!childBone) continue

    const bonePosition = bone.getWorldPosition(new THREE.Vector3())
    const childPosition = childBone.getWorldPosition(new THREE.Vector3())
    const restDirection = childPosition.sub(bonePosition).normalize()
    retargetBones.set(name, {
      bone,
      restLocalQuaternion: bone.quaternion.clone(),
      restQuaternionInRoot: inverseRootQuaternion.clone().multiply(bone.getWorldQuaternion(new THREE.Quaternion())),
      restDirectionInRoot: restDirection.applyQuaternion(inverseRootQuaternion),
    })
  }
}

function rotateBoneToward(name: string, directionInRoot: THREE.Vector3, strength: number): void {
  const restPose = retargetBones.get(name)
  if (!restPose || !playerAvatar || directionInRoot.lengthSq() < 0.0001) return

  const rootWorldQuaternion = playerAvatar.getWorldQuaternion(new THREE.Quaternion())
  const restDirectionWorld = restPose.restDirectionInRoot.clone().applyQuaternion(rootWorldQuaternion).normalize()
  const targetDirectionWorld = directionInRoot.clone().applyQuaternion(rootWorldQuaternion).normalize()
  const correction = new THREE.Quaternion().setFromUnitVectors(restDirectionWorld, targetDirectionWorld)
  const restWorldQuaternion = rootWorldQuaternion.clone().multiply(restPose.restQuaternionInRoot)
  const targetWorldQuaternion = correction.multiply(restWorldQuaternion)
  const blendedWorldQuaternion = restWorldQuaternion.slerp(targetWorldQuaternion, strength)
  const parentWorldQuaternion = restPose.bone.parent!.getWorldQuaternion(new THREE.Quaternion())

  restPose.bone.quaternion.copy(parentWorldQuaternion.invert().multiply(blendedWorldQuaternion))
  restPose.bone.updateWorldMatrix(false, true)
}

function poseToAvatarSpace(point: PosePoint, shoulderCenter: PosePoint, shoulderWidth: number): THREE.Vector3 {
  return new THREE.Vector3(
    (shoulderCenter.x - point.x) / shoulderWidth,
    (shoulderCenter.y - point.y) / shoulderWidth,
    (point.z - shoulderCenter.z) / shoulderWidth,
  )
}

function applyUpperBodyPose(): void {
  if (!trackingActive || !playerAvatar || !latestPose || retargetBones.size === 0) return

  for (const restPose of retargetBones.values()) {
    restPose.bone.quaternion.copy(restPose.restLocalQuaternion)
  }
  playerAvatar.updateWorldMatrix(true, true)

  const { leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip, shoulderWidth } = latestPose
  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: (leftShoulder.z + rightShoulder.z) / 2,
  }
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: (leftHip.z + rightHip.z) / 2,
  }
  const leftShoulderPoint = poseToAvatarSpace(leftShoulder, shoulderCenter, shoulderWidth)
  const rightShoulderPoint = poseToAvatarSpace(rightShoulder, shoulderCenter, shoulderWidth)
  const leftElbowPoint = poseToAvatarSpace(leftElbow, shoulderCenter, shoulderWidth)
  const rightElbowPoint = poseToAvatarSpace(rightElbow, shoulderCenter, shoulderWidth)
  const leftWristPoint = poseToAvatarSpace(leftWrist, shoulderCenter, shoulderWidth)
  const rightWristPoint = poseToAvatarSpace(rightWrist, shoulderCenter, shoulderWidth)
  const hipCenterPoint = poseToAvatarSpace(hipCenter, shoulderCenter, shoulderWidth)

  rotateBoneToward('mixamorigSpine', hipCenterPoint.clone().negate(), 0.45)
  rotateBoneToward('mixamorigLeftArm', leftElbowPoint.sub(leftShoulderPoint), 0.9)
  rotateBoneToward('mixamorigLeftForeArm', leftWristPoint.sub(leftElbowPoint), 0.9)
  rotateBoneToward('mixamorigRightArm', rightElbowPoint.sub(rightShoulderPoint), 0.9)
  rotateBoneToward('mixamorigRightForeArm', rightWristPoint.sub(rightElbowPoint), 0.9)
}

async function loadPlayerAvatar(): Promise<void> {
  try {
    const [avatar, hook] = await Promise.all([
      fbxLoader.loadAsync('/assets/mixamo/character/y-bot.fbx'),
      fbxLoader.loadAsync('/assets/mixamo/animations/hook.fbx'),
    ])
    avatar.scale.setScalar(0.01)
    avatar.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        object.castShadow = true
        object.receiveShadow = true
      }
    })
    avatar.position.copy(player.position)
    avatar.rotation.copy(player.rotation)
    scene.add(avatar)
    scene.remove(player)
    playerAvatar = avatar
    avatar.updateWorldMatrix(true, true)
    cacheRetargetBones(avatar)

    if (hook.animations[0]) {
      playerMixer = new THREE.AnimationMixer(avatar)
      playerMixer.clipAction(hook.animations[0]).play()
    }
  } catch (error) {
    console.error('Unable to load Y Bot or Hook animation', error)
  }
}

void loadPlayerAvatar()

window.addEventListener('keydown', (event) => {
  keys.add(event.key.toLowerCase())
  if (event.key.toLowerCase() === 'r') {
    player.position.set(0, 0, 2.2)
    playerStamina = 100
  }
})
window.addEventListener('keyup', (event) => keys.delete(event.key.toLowerCase()))

function setTrackingStatus(message: string, active = false): void {
  trackingStatus.textContent = message
  trackingDot.classList.toggle('is-active', active)
}

async function startCamera(): Promise<void> {
  cameraButton.disabled = true
  setTrackingStatus('carregando modelo...')

  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm',
    )
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    })
    webcam.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
    await webcam.play()
    trackingActive = true
    cameraGloves.visible = true
    playerMixer?.stopAllAction()
    cameraButton.textContent = 'Camera ativa'
    setTrackingStatus('procurando postura...')
  } catch (error) {
    console.error('Unable to start camera pose tracking', error)
    cameraButton.disabled = false
    setTrackingStatus('camera indisponivel')
  }
}

cameraButton.addEventListener('click', () => void startCamera())

function updateTrackedGloves(): void {
  if (!trackingActive || !poseLandmarker || webcam.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
  if (webcam.currentTime === lastVideoTime) return

  lastVideoTime = webcam.currentTime
  const result = poseLandmarker.detectForVideo(webcam, performance.now())
  const landmarks = result.landmarks[0]
  if (!landmarks) {
    setTrackingStatus('corpo fora do quadro')
    return
  }

  const leftShoulder = landmarks[11]
  const rightShoulder = landmarks[12]
  const leftWrist = landmarks[15]
  const rightWrist = landmarks[16]
  const leftElbow = landmarks[13]
  const rightElbow = landmarks[14]
  const leftHip = landmarks[23]
  const rightHip = landmarks[24]
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x)
  const visibility = Math.min(leftWrist.visibility ?? 0, rightWrist.visibility ?? 0)

  if (shoulderWidth < 0.08 || visibility < 0.45) {
    setTrackingStatus('mostre ombros e maos')
    return
  }

  const mapWrist = (wrist: typeof leftWrist, shoulder: typeof leftShoulder, side: number): THREE.Vector3 => {
    const horizontal = (wrist.x - shoulder.x) / shoulderWidth
    const vertical = (shoulder.y - wrist.y) / shoulderWidth
    const depth = THREE.MathUtils.clamp((wrist.z - shoulder.z) / shoulderWidth, -0.45, 0.85)
    return new THREE.Vector3(
      THREE.MathUtils.clamp(side * 0.54 - horizontal * 1.25, -1.35, 1.35),
      THREE.MathUtils.clamp(1.55 + vertical * 1.1, 0.68, 2.6),
      THREE.MathUtils.clamp(0.08 + depth * 0.8, -0.5, 1.15),
    )
  }

  trackedGloveTargets.left.copy(mapWrist(leftWrist, leftShoulder, -1))
  trackedGloveTargets.right.copy(mapWrist(rightWrist, rightShoulder, 1))
  latestPose = { leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip, shoulderWidth }
  setTrackingStatus('rastreamento ativo', true)
}

function punch(fighter: THREE.Group, gloveName: string, amount: number): void {
  const glove = fighter.getObjectByName(gloveName)!
  glove.position.z = THREE.MathUtils.damp(glove.position.z, amount, 18, 1 / 60)
}

function updateHud(): void {
  document.querySelector('#player-stamina')!.textContent = Math.round(playerStamina).toString()
  document.querySelector('#opponent-stamina')!.textContent = Math.round(opponentStamina).toString()
  ;(document.querySelector('#player-meter') as HTMLElement).style.width = `${playerStamina}%`
  ;(document.querySelector('#opponent-meter') as HTMLElement).style.width = `${opponentStamina}%`
}

function animate(time: number): void {
  const delta = Math.min((time - lastTime) / 1000, 0.05)
  lastTime = time
  const moveSpeed = playerStamina < 25 ? 1.15 : 1.35
  const movement = new THREE.Vector3(
    Number(keys.has('d')) - Number(keys.has('a')),
    0,
    Number(keys.has('s')) - Number(keys.has('w')),
  )
  if (movement.lengthSq()) {
    movement.normalize().multiplyScalar(moveSpeed * delta)
    player.position.add(movement)
    player.position.x = THREE.MathUtils.clamp(player.position.x, -3.65, 3.65)
    player.position.z = THREE.MathUtils.clamp(player.position.z, -3.65, 3.65)
    playerStamina = Math.max(0, playerStamina - 3 * delta)
  } else {
    playerStamina = Math.min(100, playerStamina + 12 * delta)
  }

  const playerToOpponent = opponent.position.clone().sub(player.position)
  player.rotation.y = Math.atan2(playerToOpponent.x, playerToOpponent.z)
  opponent.rotation.y = Math.atan2(-playerToOpponent.x, -playerToOpponent.z)
  opponent.position.x = Math.sin(time / 1250) * 1.3
  opponentStamina = 72 + Math.sin(time / 1900) * 12

  if (playerAvatar) {
    playerAvatar.position.copy(player.position)
    playerAvatar.rotation.copy(player.rotation)
  }

  updateTrackedGloves()
  if (trackingActive) {
    cameraGloves.position.copy(player.position)
    cameraGloves.rotation.copy(player.rotation)
    trackedLeftGlove.position.lerp(trackedGloveTargets.left, 0.22)
    trackedRightGlove.position.lerp(trackedGloveTargets.right, 0.22)
    applyUpperBodyPose()
  } else {
    punch(player, 'left-glove', keys.has('j') && playerStamina >= 5 ? 0.9 : 0.08)
    punch(player, 'right-glove', keys.has('k') && playerStamina >= 7 ? 1.05 : 0.08)
  }
  playerMixer?.update(delta)
  updateHud()
  renderer.render(scene, camera)
  requestAnimationFrame(animate)
}

function resize(): void {
  const { width, height } = canvas.getBoundingClientRect()
  renderer.setSize(width, height, false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
}

new ResizeObserver(resize).observe(canvas)
resize()
requestAnimationFrame(animate)
