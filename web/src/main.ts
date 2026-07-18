import './style.css'
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision'
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
      <canvas id="pose-overlay" aria-hidden="true"></canvas>
      <div class="camera-panel__status"><i id="tracking-dot"></i><span id="tracking-status">camera desligada</span></div>
      <button id="camera-button" type="button">Ativar camera</button>
    </section>
    <section class="comparison-legend" aria-label="Comparação dos rastreadores">
      <span><i class="comparison-legend__pose"></i>ESQUERDA: CORPO</span>
      <span><i class="comparison-legend__fused"></i>DIREITA: MAOS -&gt; IK</span>
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
camera.position.set(1.15, 2.55, 6.4)
camera.lookAt(0, 1.35, 0)
let cameraOrbitYaw = 0
let cameraOrbitPitch = 0
let cameraDragPointer: { id: number; x: number; y: number } | undefined

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

const PLAYER_START_POSITION = new THREE.Vector3(0, 0, 3.2)
const PLAYER_START_ROTATION_Y = Math.PI

const player = createFighter('#1f8c87', '#e7b96a')
player.position.copy(PLAYER_START_POSITION)
player.rotation.y = PLAYER_START_ROTATION_Y
scene.add(player)

type PosePoint = { x: number; y: number; z: number }
type ImagePoint = PosePoint & { visibility?: number }
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
  childBone: THREE.Bone
  restLocalQuaternion: THREE.Quaternion
  restQuaternionInRoot: THREE.Quaternion
  restDirectionInRoot: THREE.Vector3
  restPalmNormalInRoot: THREE.Vector3
}
type AvatarRig = {
  avatar: THREE.Group
  bones: Map<string, BoneRestPose>
}
type TrackedHands = {
  left?: readonly ImagePoint[]
  right?: readonly ImagePoint[]
}
type HandDepths = {
  left?: number
  right?: number
}
type HandSide = 'left' | 'right'

let poseAvatar: AvatarRig | undefined
let playerAvatar: AvatarRig | undefined
let latestPoseOnly: TrackedPose | undefined
let latestPoseWithHandDepth: TrackedPose | undefined

const keys = new Set<string>()
const webcam = document.querySelector<HTMLVideoElement>('#webcam')!
const poseOverlay = document.querySelector<HTMLCanvasElement>('#pose-overlay')!
const poseContext = poseOverlay.getContext('2d')!
const cameraButton = document.querySelector<HTMLButtonElement>('#camera-button')!
const trackingStatus = document.querySelector<HTMLSpanElement>('#tracking-status')!
const trackingDot = document.querySelector<HTMLElement>('#tracking-dot')!
let playerStamina = 100
let opponentStamina = 100
let lastTime = performance.now()
const fbxLoader = new FBXLoader()
let poseLandmarker: PoseLandmarker | undefined
let handLandmarker: HandLandmarker | undefined
let lastVideoTime = -1
let trackingActive = false
let latestHandLandmarks: readonly (readonly ImagePoint[])[] = []
let latestHandLabels: Array<'Left' | 'Right' | undefined> = []
let latestTrackedHands: TrackedHands = {}
let latestHandDepths: HandDepths = {}
const handDepthCalibration: Record<HandSide, { wrist: ImagePoint; palmSpan: number; palmDepth: number; poseDepth: number } | undefined> = {
  left: undefined,
  right: undefined,
}

canvas.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return
  cameraDragPointer = { id: event.pointerId, x: event.clientX, y: event.clientY }
  canvas.setPointerCapture(event.pointerId)
  canvas.classList.add('is-dragging')
})

canvas.addEventListener('pointermove', (event) => {
  if (!cameraDragPointer || event.pointerId !== cameraDragPointer.id) return
  cameraOrbitYaw -= (event.clientX - cameraDragPointer.x) * 0.008
  cameraOrbitPitch = THREE.MathUtils.clamp(cameraOrbitPitch - (event.clientY - cameraDragPointer.y) * 0.006, -0.55, 0.55)
  cameraDragPointer.x = event.clientX
  cameraDragPointer.y = event.clientY
})

function stopCameraDrag(event: PointerEvent): void {
  if (!cameraDragPointer || event.pointerId !== cameraDragPointer.id) return
  cameraDragPointer = undefined
  canvas.classList.remove('is-dragging')
}

canvas.addEventListener('pointerup', stopCameraDrag)
canvas.addEventListener('pointercancel', stopCameraDrag)

function firstChildBone(bone: THREE.Bone): THREE.Bone | undefined {
  return bone.children.find(
    (child): child is THREE.Bone => child instanceof THREE.Bone && child.name !== bone.name,
  )
}

function preferredChildBone(bone: THREE.Bone, boneName: string): THREE.Bone | undefined {
  // Mixamo Hand lists Thumb first; using it as rest axis makes palms look sideways.
  if (boneName.endsWith('Hand')) {
    const side = boneName.includes('Left') ? 'Left' : 'Right'
    const middle = bone.getObjectByName(`mixamorig${side}HandMiddle1`)
    if (middle instanceof THREE.Bone) return middle
  }
  return firstChildBone(bone)
}

function cacheRetargetBones(avatar: THREE.Group): Map<string, BoneRestPose> {
  const retargetBones = new Map<string, BoneRestPose>()
  const handBoneNames = ['Left', 'Right'].flatMap((side) => [
    `mixamorig${side}Hand`,
    ...['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].flatMap((finger) =>
      [1, 2, 3].map((segment) => `mixamorig${side}Hand${finger}${segment}`),
    ),
  ])
  const boneNames = [
    'mixamorigSpine', 'mixamorigLeftArm', 'mixamorigLeftForeArm', 'mixamorigRightArm', 'mixamorigRightForeArm', ...handBoneNames,
  ]
  const rootWorldQuaternion = avatar.getWorldQuaternion(new THREE.Quaternion())
  const inverseRootQuaternion = rootWorldQuaternion.clone().invert()

  for (const name of boneNames) {
    const bone = avatar.getObjectByName(name)
    if (!(bone instanceof THREE.Bone)) continue
    const childBone = preferredChildBone(bone, name)
    if (!childBone) continue

    const bonePosition = bone.getWorldPosition(new THREE.Vector3())
    const childPosition = childBone.getWorldPosition(new THREE.Vector3())
    const restDirection = childPosition.sub(bonePosition).normalize()
    let restPalmNormalInRoot = new THREE.Vector3(0, -1, 0)
    if (name.endsWith('Hand')) {
      const side = name.includes('Left') ? 'Left' : 'Right'
      const index = avatar.getObjectByName(`mixamorig${side}HandIndex1`)
      const pinky = avatar.getObjectByName(`mixamorig${side}HandPinky1`)
      if (index instanceof THREE.Bone && pinky instanceof THREE.Bone) {
        const indexPosition = index.getWorldPosition(new THREE.Vector3())
        const pinkyPosition = pinky.getWorldPosition(new THREE.Vector3())
        const acrossPalm = indexPosition.sub(pinkyPosition).normalize()
        const restPalmNormalWorld = new THREE.Vector3().crossVectors(restDirection, acrossPalm).normalize()
        if (restPalmNormalWorld.lengthSq() > 0.0001) {
          restPalmNormalInRoot = restPalmNormalWorld.applyQuaternion(inverseRootQuaternion)
        }
      }
    }
    retargetBones.set(name, {
      bone,
      childBone,
      restLocalQuaternion: bone.quaternion.clone(),
      restQuaternionInRoot: inverseRootQuaternion.clone().multiply(bone.getWorldQuaternion(new THREE.Quaternion())),
      restDirectionInRoot: restDirection.applyQuaternion(inverseRootQuaternion),
      restPalmNormalInRoot,
    })
  }
  return retargetBones
}

function rotateBoneToward(rig: AvatarRig, name: string, directionInRoot: THREE.Vector3, strength: number): void {
  const restPose = rig.bones.get(name)
  if (!restPose || directionInRoot.lengthSq() < 0.0001) return

  const rootWorldQuaternion = rig.avatar.getWorldQuaternion(new THREE.Quaternion())
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

function orientBoneWithPalm(
  rig: AvatarRig,
  name: string,
  forwardInRoot: THREE.Vector3,
  palmNormalInRoot: THREE.Vector3,
  strength: number,
): void {
  const restPose = rig.bones.get(name)
  if (!restPose || forwardInRoot.lengthSq() < 0.0001) return

  const rootWorldQuaternion = rig.avatar.getWorldQuaternion(new THREE.Quaternion())
  const restDirectionWorld = restPose.restDirectionInRoot.clone().applyQuaternion(rootWorldQuaternion).normalize()
  const targetDirectionWorld = forwardInRoot.clone().normalize().applyQuaternion(rootWorldQuaternion).normalize()
  const align = new THREE.Quaternion().setFromUnitVectors(restDirectionWorld, targetDirectionWorld)
  const restWorldQuaternion = rootWorldQuaternion.clone().multiply(restPose.restQuaternionInRoot)
  let targetWorldQuaternion = align.clone().multiply(restWorldQuaternion)

  const alignedPalmWorld = restPose.restPalmNormalInRoot
    .clone()
    .applyQuaternion(rootWorldQuaternion)
    .applyQuaternion(align)
    .normalize()
  const desiredPalmWorld = palmNormalInRoot.clone().normalize().applyQuaternion(rootWorldQuaternion).normalize()
  const alignedPlane = alignedPalmWorld.addScaledVector(targetDirectionWorld, -alignedPalmWorld.dot(targetDirectionWorld))
  const desiredPlane = desiredPalmWorld.addScaledVector(targetDirectionWorld, -desiredPalmWorld.dot(targetDirectionWorld))
  if (alignedPlane.lengthSq() > 0.0001 && desiredPlane.lengthSq() > 0.0001) {
    const twist = new THREE.Quaternion().setFromUnitVectors(alignedPlane.normalize(), desiredPlane.normalize())
    targetWorldQuaternion = twist.multiply(targetWorldQuaternion)
  }

  const blendedWorldQuaternion = restWorldQuaternion.clone().slerp(targetWorldQuaternion, strength)
  const parentWorldQuaternion = restPose.bone.parent!.getWorldQuaternion(new THREE.Quaternion())
  restPose.bone.quaternion.copy(parentWorldQuaternion.invert().multiply(blendedWorldQuaternion))
  restPose.bone.updateWorldMatrix(false, true)
}

function solveArm(
  rig: AvatarRig,
  upperArmName: string,
  foreArmName: string,
  shoulder: THREE.Vector3,
  elbow: THREE.Vector3,
  wrist: THREE.Vector3,
): void {
  const upperArm = rig.bones.get(upperArmName)
  const foreArm = rig.bones.get(foreArmName)
  if (!upperArm || !foreArm) return

  const shoulderWorld = upperArm.bone.getWorldPosition(new THREE.Vector3())
  const elbowWorld = foreArm.bone.getWorldPosition(new THREE.Vector3())
  const wristWorld = foreArm.childBone.getWorldPosition(new THREE.Vector3())
  const upperArmLength = shoulderWorld.distanceTo(elbowWorld)
  const foreArmLength = elbowWorld.distanceTo(wristWorld)
  const sourceUpperArm = elbow.clone().sub(shoulder)
  const sourceForeArm = wrist.clone().sub(elbow)
  const sourceWrist = wrist.clone().sub(shoulder)
  const sourceReach = sourceUpperArm.length() + sourceForeArm.length()
  if (upperArmLength < 0.001 || foreArmLength < 0.001 || sourceReach < 0.001) return

  const reach = upperArmLength + foreArmLength
  const minimumReach = Math.abs(upperArmLength - foreArmLength) + 0.001
  const targetDistance = THREE.MathUtils.clamp(sourceWrist.length() / sourceReach * reach, minimumReach, reach - 0.001)
  const targetDirection = sourceWrist.normalize()
  const targetWorld = shoulderWorld.clone().addScaledVector(targetDirection.applyQuaternion(rig.avatar.getWorldQuaternion(new THREE.Quaternion())), targetDistance)
  const shoulderToTarget = targetWorld.clone().sub(shoulderWorld)
  const shoulderToTargetLength = shoulderToTarget.length()
  if (shoulderToTargetLength < 0.001) return

  const axis = shoulderToTarget.normalize()
  const sourceElbowDirection = sourceUpperArm.normalize().applyQuaternion(rig.avatar.getWorldQuaternion(new THREE.Quaternion()))
  const elbowPlane = sourceElbowDirection.addScaledVector(axis, -sourceElbowDirection.dot(axis))
  if (elbowPlane.lengthSq() < 0.0001) elbowPlane.copy(new THREE.Vector3(0, 1, 0).cross(axis))
  elbowPlane.normalize()

  const elbowAlongAxis = (upperArmLength ** 2 - foreArmLength ** 2 + shoulderToTargetLength ** 2) / (2 * shoulderToTargetLength)
  const elbowOffset = Math.sqrt(Math.max(0, upperArmLength ** 2 - elbowAlongAxis ** 2))
  const solvedElbowWorld = shoulderWorld.clone()
    .addScaledVector(axis, elbowAlongAxis)
    .addScaledVector(elbowPlane, elbowOffset)
  const inverseRootQuaternion = rig.avatar.getWorldQuaternion(new THREE.Quaternion()).invert()

  rotateBoneToward(rig, upperArmName, solvedElbowWorld.sub(shoulderWorld).applyQuaternion(inverseRootQuaternion), 1)
  rotateBoneToward(rig, foreArmName, targetWorld.sub(foreArm.bone.getWorldPosition(new THREE.Vector3())).applyQuaternion(inverseRootQuaternion), 1)
}

function poseToAvatarSpace(point: PosePoint, shoulderCenter: PosePoint, shoulderWidth: number): THREE.Vector3 {
  return new THREE.Vector3(
    (point.x - shoulderCenter.x) / shoulderWidth,
    (shoulderCenter.y - point.y) / shoulderWidth,
    (shoulderCenter.z - point.z) / shoulderWidth,
  )
}

function applyUpperBodyPose(rig: AvatarRig | undefined, pose: TrackedPose | undefined): void {
  if (!trackingActive || !rig || !pose || rig.bones.size === 0) return

  for (const restPose of rig.bones.values()) {
    restPose.bone.quaternion.copy(restPose.restLocalQuaternion)
  }
  rig.avatar.updateWorldMatrix(true, true)

  const { leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist, leftHip, rightHip, shoulderWidth } = pose
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

  rotateBoneToward(rig, 'mixamorigSpine', hipCenterPoint.clone().negate(), 0.45)
  solveArm(rig, 'mixamorigLeftArm', 'mixamorigLeftForeArm', leftShoulderPoint, leftElbowPoint, leftWristPoint)
  solveArm(rig, 'mixamorigRightArm', 'mixamorigRightForeArm', rightShoulderPoint, rightElbowPoint, rightWristPoint)
}

const fingerLandmarks = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
} as const

/** MediaPipe image deltas → avatar root space. Z flips so "toward camera" is character forward. */
function mediaPipeDeltaToAvatar(from: ImagePoint, to: ImagePoint): THREE.Vector3 {
  return new THREE.Vector3(
    to.x - from.x,
    from.y - to.y,
    from.z - to.z,
  )
}

function handPalmBasis(side: HandSide, hand: readonly ImagePoint[]): { forward: THREE.Vector3; palmNormal: THREE.Vector3 } {
  const wrist = hand[0]
  const index = hand[5]
  const middle = hand[9]
  const pinky = hand[17]
  const forward = mediaPipeDeltaToAvatar(wrist, middle)
  if (forward.lengthSq() < 1e-8) forward.set(0, 0, 1)
  forward.normalize()

  // Both hands use the same palm winding after conversion into avatar space.
  const across = mediaPipeDeltaToAvatar(pinky, index)
  let palmNormal = new THREE.Vector3().crossVectors(forward, across)
  if (palmNormal.lengthSq() < 1e-8) {
    palmNormal = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(side === 'left' ? -1 : 1, 0, 0))
  }
  if (palmNormal.lengthSq() < 1e-8) palmNormal.set(0, 0, 1)
  palmNormal.normalize()

  const correctedAcross = new THREE.Vector3().crossVectors(palmNormal, forward).normalize()
  palmNormal.crossVectors(forward, correctedAcross).normalize()
  return { forward, palmNormal }
}

function handForwardOffset(side: HandSide, hand: readonly ImagePoint[], poseDepth: number): number {
  const palmSpan = Math.hypot(hand[5].x - hand[17].x, hand[5].y - hand[17].y)
  const palmDepth = (hand[5].z + hand[9].z + hand[13].z + hand[17].z) / 4
  const calibration = handDepthCalibration[side] ?? { wrist: { ...hand[0] }, palmSpan, palmDepth, poseDepth }
  handDepthCalibration[side] ??= calibration

  const scaleAdvance = (palmSpan / Math.max(calibration.palmSpan, 1e-4) - 1) * 3.8
  const landmarkAdvance = (calibration.palmDepth - palmDepth) * 2.6
  // This is deliberately amplified for the A/B test: the body owns X/Y, while both hand-depth signals visibly own Z.
  return THREE.MathUtils.clamp(scaleAdvance + landmarkAdvance, -0.75, 2.2)
}

function applyHandDepthToPose(pose: TrackedPose, hands: TrackedHands): TrackedPose {
  const leftOffset = hands.left ? handForwardOffset('left', hands.left, latestHandDepths.left ?? 0) : 0
  const rightOffset = hands.right ? handForwardOffset('right', hands.right, latestHandDepths.right ?? 0) : 0
  return {
    ...pose,
    leftWrist: { ...pose.leftWrist, z: pose.leftWrist.z - leftOffset * pose.shoulderWidth },
    rightWrist: { ...pose.rightWrist, z: pose.rightWrist.z - rightOffset * pose.shoulderWidth },
  }
}

function applyFingerPose(rig: AvatarRig, side: HandSide, hand: readonly ImagePoint[]): void {
  const rigSide = side === 'left' ? 'Left' : 'Right'
  const { forward, palmNormal } = handPalmBasis(side, hand)
  orientBoneWithPalm(rig, `mixamorig${rigSide}Hand`, forward, palmNormal, 0.92)

  for (const [finger, landmarks] of Object.entries(fingerLandmarks)) {
    for (let segment = 0; segment < 3; segment += 1) {
      const from = hand[landmarks[segment]]
      const to = hand[landmarks[segment + 1]]
      if (!from || !to) continue
      rotateBoneToward(rig, `mixamorig${rigSide}Hand${finger}${segment + 1}`, mediaPipeDeltaToAvatar(from, to), 0.85)
    }
  }
}

function applyHandDrivenPose(
  rig: AvatarRig | undefined,
  hands: TrackedHands,
): void {
  if (!trackingActive || !rig || (!hands.left && !hands.right)) return
  for (const [side, hand] of Object.entries(hands) as Array<[HandSide, readonly ImagePoint[] | undefined]>) {
    if (!hand) continue
    applyFingerPose(rig, side, hand)
  }
}

function applyGuardPose(rig: AvatarRig | undefined): void {
  if (!rig || rig.bones.size === 0) return

  for (const restPose of rig.bones.values()) {
    restPose.bone.quaternion.copy(restPose.restLocalQuaternion)
  }
  rig.avatar.updateWorldMatrix(true, true)

  rotateBoneToward(rig, 'mixamorigLeftArm', new THREE.Vector3(-0.5, 0.05, 0.7), 1)
  rotateBoneToward(rig, 'mixamorigLeftForeArm', new THREE.Vector3(0, 0.6, 0.55), 1)
  rotateBoneToward(rig, 'mixamorigRightArm', new THREE.Vector3(0.5, 0.05, 0.7), 1)
  rotateBoneToward(rig, 'mixamorigRightForeArm', new THREE.Vector3(0, 0.6, 0.55), 1)
}

function prepareAvatar(avatar: THREE.Group, offsetX: number): AvatarRig {
  avatar.scale.setScalar(0.01)
  avatar.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true
      object.receiveShadow = true
    }
  })
  avatar.position.copy(player.position).add(new THREE.Vector3(offsetX, 0, 0))
  avatar.rotation.copy(player.rotation)
  scene.add(avatar)
  avatar.updateWorldMatrix(true, true)
  return { avatar, bones: cacheRetargetBones(avatar) }
}

async function loadPlayerAvatars(): Promise<void> {
  try {
    const [poseModel, fusedModel] = await Promise.all([
      fbxLoader.loadAsync('/assets/mixamo/character/y-bot.fbx'),
      fbxLoader.loadAsync('/assets/mixamo/character/y-bot.fbx'),
    ])
    scene.remove(player)
    poseAvatar = prepareAvatar(poseModel, -1.2)
    playerAvatar = prepareAvatar(fusedModel, 1.2)
  } catch (error) {
    console.error('Unable to load Y Bots', error)
  }
}

void loadPlayerAvatars()

window.addEventListener('keydown', (event) => {
  keys.add(event.key.toLowerCase())
  if (event.key.toLowerCase() === 'r') {
    player.position.copy(PLAYER_START_POSITION)
    player.rotation.y = PLAYER_START_ROTATION_Y
    playerStamina = 100
  }
})
window.addEventListener('keyup', (event) => keys.delete(event.key.toLowerCase()))

function setTrackingStatus(message: string, active = false): void {
  trackingStatus.textContent = message
  trackingDot.classList.toggle('is-active', active)
}

const poseConnections: ReadonlyArray<readonly [number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
]
const handConnections: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17],
]

function drawPoseOverlay(
  landmarks: readonly PosePoint[] | undefined,
  hands: readonly (readonly ImagePoint[])[],
): void {
  const width = webcam.clientWidth
  const height = webcam.clientHeight
  if (poseOverlay.width !== width || poseOverlay.height !== height) {
    poseOverlay.width = width
    poseOverlay.height = height
  }
  poseContext.clearRect(0, 0, width, height)
  if (!landmarks?.length || !webcam.videoWidth || !webcam.videoHeight) return

  const videoScale = Math.max(width / webcam.videoWidth, height / webcam.videoHeight)
  const renderedWidth = webcam.videoWidth * videoScale
  const renderedHeight = webcam.videoHeight * videoScale
  const offsetX = (width - renderedWidth) / 2
  const offsetY = (height - renderedHeight) / 2
  const pointFor = (landmark: PosePoint): [number, number] => [
    offsetX + landmark.x * renderedWidth,
    offsetY + landmark.y * renderedHeight,
  ]

  poseContext.lineCap = 'round'
  poseContext.lineWidth = 2
  poseContext.strokeStyle = '#42d8ae'
  for (const [firstIndex, secondIndex] of poseConnections) {
    const first = landmarks[firstIndex]
    const second = landmarks[secondIndex]
    if (!first || !second) continue
    const [firstX, firstY] = pointFor(first)
    const [secondX, secondY] = pointFor(second)
    poseContext.beginPath()
    poseContext.moveTo(firstX, firstY)
    poseContext.lineTo(secondX, secondY)
    poseContext.stroke()
  }

  for (const index of new Set(poseConnections.flat())) {
    const landmark = landmarks[index]
    if (!landmark) continue
    const [x, y] = pointFor(landmark)
    poseContext.fillStyle = (landmark as typeof landmark & { visibility?: number }).visibility ?? 0 >= 0.45 ? '#42d8ae' : '#df9a4e'
    poseContext.beginPath()
    poseContext.arc(x, y, 3, 0, Math.PI * 2)
    poseContext.fill()
  }

  poseContext.strokeStyle = '#f2d08d'
  poseContext.lineWidth = 1.5
  for (const hand of hands) {
    for (const [firstIndex, secondIndex] of handConnections) {
      const first = hand[firstIndex]
      const second = hand[secondIndex]
      if (!first || !second) continue
      const [firstX, firstY] = pointFor(first)
      const [secondX, secondY] = pointFor(second)
      poseContext.beginPath()
      poseContext.moveTo(firstX, firstY)
      poseContext.lineTo(secondX, secondY)
      poseContext.stroke()
    }
    poseContext.fillStyle = '#f2d08d'
    for (const landmark of hand) {
      if (!landmark) continue
      const [x, y] = pointFor(landmark)
      poseContext.beginPath()
      poseContext.arc(x, y, 1.75, 0, Math.PI * 2)
      poseContext.fill()
    }
  }
}

function assignHandsToSides(imageLandmarks: readonly ImagePoint[]): TrackedHands {
  const assigned: TrackedHands = {}
  const unused: number[] = []

  for (let index = 0; index < latestHandLandmarks.length; index += 1) {
    const label = latestHandLabels[index]
    const hand = latestHandLandmarks[index]
    if (!hand) continue
    if (label === 'Left' && !assigned.left) {
      assigned.left = hand
      continue
    }
    if (label === 'Right' && !assigned.right) {
      assigned.right = hand
      continue
    }
    unused.push(index)
  }

  const nearestUnused = (wrist: ImagePoint): number => unused.reduce((nearestIndex, index) => {
    if (nearestIndex === -1) return index
    const hand = latestHandLandmarks[index]
    const nearest = latestHandLandmarks[nearestIndex]
    if (!hand?.[0] || !nearest?.[0]) return nearestIndex
    const distance = Math.hypot(hand[0].x - wrist.x, hand[0].y - wrist.y)
    const nearestDistance = Math.hypot(nearest[0].x - wrist.x, nearest[0].y - wrist.y)
    return distance < nearestDistance ? index : nearestIndex
  }, -1)

  if (!assigned.left) {
    const leftIndex = nearestUnused(imageLandmarks[15])
    if (leftIndex >= 0) {
      assigned.left = latestHandLandmarks[leftIndex]
      unused.splice(unused.indexOf(leftIndex), 1)
    }
  }
  if (!assigned.right) {
    const rightIndex = nearestUnused(imageLandmarks[16])
    if (rightIndex >= 0) assigned.right = latestHandLandmarks[rightIndex]
  }

  return assigned
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
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    })
    webcam.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    })
    await webcam.play()
    handDepthCalibration.left = undefined
    handDepthCalibration.right = undefined
    trackingActive = true
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
  const timestamp = performance.now()
  const result = poseLandmarker.detectForVideo(webcam, timestamp)
  if (handLandmarker) {
    const handResult = handLandmarker.detectForVideo(webcam, timestamp)
    latestHandLandmarks = handResult.landmarks as readonly (readonly ImagePoint[])[]
    latestHandLabels = handResult.handednesses.map((categories) => {
      const name = categories[0]?.categoryName
      return name === 'Left' || name === 'Right' ? name : undefined
    })
  }
  const imageLandmarks = result.landmarks[0]
  const worldLandmarks = result.worldLandmarks[0]
  drawPoseOverlay(imageLandmarks, latestHandLandmarks)
  if (!imageLandmarks || !worldLandmarks) {
    setTrackingStatus('corpo fora do quadro')
    return
  }

  const leftShoulder = worldLandmarks[11]
  const rightShoulder = worldLandmarks[12]
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x)
  const poseLeftWrist = worldLandmarks[15]
  const poseRightWrist = worldLandmarks[16]
  const leftElbow = worldLandmarks[13]
  const rightElbow = worldLandmarks[14]
  const leftHip = worldLandmarks[23]
  const rightHip = worldLandmarks[24]
  latestTrackedHands = assignHandsToSides(imageLandmarks)
  const shoulderDepth = (leftShoulder.z + rightShoulder.z) / 2
  latestHandDepths = {
    left: (shoulderDepth - poseLeftWrist.z) / shoulderWidth,
    right: (shoulderDepth - poseRightWrist.z) / shoulderWidth,
  }
  const torsoVisibility = Math.min(
    imageLandmarks[11].visibility ?? 0,
    imageLandmarks[12].visibility ?? 0,
    imageLandmarks[23].visibility ?? 0,
    imageLandmarks[24].visibility ?? 0,
  )
  const wristVisibility = Math.min(imageLandmarks[15].visibility ?? 0, imageLandmarks[16].visibility ?? 0)

  if (shoulderWidth < 0.08 || torsoVisibility < 0.45) {
    setTrackingStatus('mostre tronco e ombros')
    return
  }
  if (wristVisibility < 0.45) {
    latestPoseOnly = undefined
    latestPoseWithHandDepth = undefined
    setTrackingStatus(latestTrackedHands.left || latestTrackedHands.right ? 'maos ativas; mostre os bracos para comparar' : 'mostre as maos para os bracos')
    return
  }

  latestPoseOnly = {
    leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist: poseLeftWrist, rightWrist: poseRightWrist, leftHip, rightHip, shoulderWidth,
  }
  latestPoseWithHandDepth = applyHandDepthToPose(latestPoseOnly, latestTrackedHands)
  setTrackingStatus(latestHandLandmarks.length ? 'corpo e maos ativos' : 'rastreamento corporal ativo', true)
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

  if (poseAvatar) {
    poseAvatar.avatar.position.copy(player.position).add(new THREE.Vector3(-1.2, 0, 0))
    poseAvatar.avatar.rotation.copy(player.rotation)
  }
  if (playerAvatar) {
    playerAvatar.avatar.position.copy(player.position).add(new THREE.Vector3(1.2, 0, 0))
    playerAvatar.avatar.rotation.copy(player.rotation)
  }

  const verticalAxis = new THREE.Vector3(0, 1, 0)
  const cameraOffset = new THREE.Vector3(-1.15, 2.05, -3.4)
    .applyAxisAngle(verticalAxis, player.rotation.y + cameraOrbitYaw)
    .applyAxisAngle(new THREE.Vector3(1, 0, 0).applyAxisAngle(verticalAxis, player.rotation.y + cameraOrbitYaw), cameraOrbitPitch)
  const cameraTarget = player.position
    .clone()
    .add(new THREE.Vector3(0, 1.25, 0.35).applyAxisAngle(new THREE.Vector3(0, 1, 0), player.rotation.y))
  camera.position.lerp(player.position.clone().add(cameraOffset), 1 - Math.exp(-7 * delta))
  camera.lookAt(cameraTarget)

  updateTrackedGloves()
  if (trackingActive && latestPoseOnly) {
    applyUpperBodyPose(poseAvatar, latestPoseOnly)
    applyUpperBodyPose(playerAvatar, latestPoseWithHandDepth)
  } else {
    applyGuardPose(poseAvatar)
    applyGuardPose(playerAvatar)
    punch(player, 'left-glove', keys.has('j') && playerStamina >= 5 ? 0.9 : 0.08)
    punch(player, 'right-glove', keys.has('k') && playerStamina >= 7 ? 1.05 : 0.08)
  }
  if (trackingActive) applyHandDrivenPose(playerAvatar, latestTrackedHands)
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
