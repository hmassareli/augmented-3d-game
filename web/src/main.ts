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
import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app")!;

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
    <footer class="hud controls">WASD move &nbsp; · &nbsp; J jab &nbsp; · &nbsp; K cross &nbsp; · &nbsp; R reset</footer>
  </main>
`;

const canvas = document.querySelector<HTMLCanvasElement>("#ring")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#15191c");
scene.fog = new THREE.Fog("#15191c", 10, 28);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(1.15, 2.55, 6.4);
camera.lookAt(0, 1.35, 0);
let cameraOrbitYaw = 0;
let cameraOrbitPitch = 0;
let cameraDragPointer: { id: number; x: number; y: number } | undefined;

scene.add(new THREE.HemisphereLight("#b8e8ef", "#27202b", 2.3));
const keyLight = new THREE.SpotLight("#fff2dc", 180, 25, Math.PI / 5, 0.45);
keyLight.position.set(-3, 9, 5);
keyLight.castShadow = true;
scene.add(keyLight);

const ring = new THREE.Group();
scene.add(ring);

const canvasFloor = new THREE.Mesh(
  new THREE.BoxGeometry(8.2, 0.25, 8.2),
  new THREE.MeshStandardMaterial({ color: "#d9d0c0", roughness: 0.86 }),
);
canvasFloor.receiveShadow = true;
ring.add(canvasFloor);

const apron = new THREE.Mesh(
  new THREE.BoxGeometry(9.3, 0.55, 9.3),
  new THREE.MeshStandardMaterial({ color: "#1e3437", roughness: 0.62 }),
);
apron.position.y = -0.38;
apron.receiveShadow = true;
ring.add(apron);

function createFighter(bodyColor: string, gloveColor: string): THREE.Group {
  const fighter = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: bodyColor,
    roughness: 0.63,
  });
  const gloveMaterial = new THREE.MeshStandardMaterial({
    color: gloveColor,
    roughness: 0.35,
    metalness: 0.05,
  });
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.46, 1.0, 8, 16),
    material,
  );
  torso.position.y = 1.2;
  torso.castShadow = true;
  fighter.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.34, 16, 12), material);
  head.position.y = 2.08;
  head.castShadow = true;
  fighter.add(head);
  for (const side of [-1, 1]) {
    const glove = new THREE.Mesh(
      new THREE.SphereGeometry(0.26, 16, 12),
      gloveMaterial,
    );
    glove.name = side < 0 ? "left-glove" : "right-glove";
    glove.position.set(side * 0.54, 1.55, 0.08);
    glove.scale.set(0.85, 1.12, 1.05);
    glove.castShadow = true;
    fighter.add(glove);
  }
  return fighter;
}

const PLAYER_START_POSITION = new THREE.Vector3(0, 0, 3.2);
const PLAYER_START_ROTATION_Y = Math.PI;

const player = createFighter("#1f8c87", "#e7b96a");
player.position.copy(PLAYER_START_POSITION);
player.rotation.y = PLAYER_START_ROTATION_Y;
scene.add(player);

type ImagePoint = PosePoint & { visibility?: number };
type TrackedHands = {
  left?: readonly ImagePoint[];
  right?: readonly ImagePoint[];
};
type HandSide = "left" | "right";

let playerAvatar: AvatarRig | undefined;
let latestPose: TrackedPose | undefined;

const keys = new Set<string>();
const webcam = document.querySelector<HTMLVideoElement>("#webcam")!;
const poseOverlay = document.querySelector<HTMLCanvasElement>("#pose-overlay")!;
const poseContext = poseOverlay.getContext("2d")!;
const cameraButton =
  document.querySelector<HTMLButtonElement>("#camera-button")!;
const trackingStatus =
  document.querySelector<HTMLSpanElement>("#tracking-status")!;
const trackingDot = document.querySelector<HTMLElement>("#tracking-dot")!;
let playerStamina = 100;
let opponentStamina = 100;
let lastTime = performance.now();
const fbxLoader = new FBXLoader();
let poseLandmarker: PoseLandmarker | undefined;
let handLandmarker: HandLandmarker | undefined;
let lastVideoTime = -1;
let trackingActive = false;
let latestHandLandmarks: readonly (readonly ImagePoint[])[] = [];
let latestHandLabels: Array<"Left" | "Right" | undefined> = [];
let latestTrackedHands: TrackedHands = {};

canvas.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  cameraDragPointer = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
  };
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("is-dragging");
});

canvas.addEventListener("pointermove", (event) => {
  if (!cameraDragPointer || event.pointerId !== cameraDragPointer.id) return;
  cameraOrbitYaw -= (event.clientX - cameraDragPointer.x) * 0.008;
  cameraOrbitPitch = THREE.MathUtils.clamp(
    cameraOrbitPitch - (event.clientY - cameraDragPointer.y) * 0.006,
    -0.55,
    0.55,
  );
  cameraDragPointer.x = event.clientX;
  cameraDragPointer.y = event.clientY;
});

function stopCameraDrag(event: PointerEvent): void {
  if (!cameraDragPointer || event.pointerId !== cameraDragPointer.id) return;
  cameraDragPointer = undefined;
  canvas.classList.remove("is-dragging");
}

canvas.addEventListener("pointerup", stopCameraDrag);
canvas.addEventListener("pointercancel", stopCameraDrag);

const fingerLandmarks = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
} as const;

/** MediaPipe image deltas → avatar root space. Z flips so "toward camera" is character forward. */
function mediaPipeDeltaToAvatar(
  from: ImagePoint,
  to: ImagePoint,
): THREE.Vector3 {
  return new THREE.Vector3(to.x - from.x, from.y - to.y, from.z - to.z);
}

function handPalmBasis(
  side: HandSide,
  hand: readonly ImagePoint[],
): { forward: THREE.Vector3; palmNormal: THREE.Vector3 } {
  const wrist = hand[0];
  const index = hand[5];
  const middle = hand[9];
  const pinky = hand[17];
  const forward = mediaPipeDeltaToAvatar(wrist, middle);
  if (forward.lengthSq() < 1e-8) forward.set(0, 0, 1);
  forward.normalize();

  // Both hands use the same palm winding after conversion into avatar space.
  const across = mediaPipeDeltaToAvatar(pinky, index);
  let palmNormal = new THREE.Vector3().crossVectors(forward, across);
  if (palmNormal.lengthSq() < 1e-8) {
    palmNormal = new THREE.Vector3().crossVectors(
      forward,
      new THREE.Vector3(side === "left" ? -1 : 1, 0, 0),
    );
  }
  if (palmNormal.lengthSq() < 1e-8) palmNormal.set(0, 0, 1);
  palmNormal.normalize();

  const correctedAcross = new THREE.Vector3()
    .crossVectors(palmNormal, forward)
    .normalize();
  palmNormal.crossVectors(forward, correctedAcross).normalize();
  return { forward, palmNormal };
}

function applyFingerPose(
  rig: AvatarRig,
  side: HandSide,
  hand: readonly ImagePoint[],
): void {
  const rigSide = side === "left" ? "Left" : "Right";
  const { forward, palmNormal } = handPalmBasis(side, hand);
  orientBoneWithPalm(rig, `mixamorig${rigSide}Hand`, forward, palmNormal, 0.92);

  for (const [finger, landmarks] of Object.entries(fingerLandmarks)) {
    for (let segment = 0; segment < 3; segment += 1) {
      const from = hand[landmarks[segment]];
      const to = hand[landmarks[segment + 1]];
      if (!from || !to) continue;
      rotateBoneToward(
        rig,
        `mixamorig${rigSide}Hand${finger}${segment + 1}`,
        mediaPipeDeltaToAvatar(from, to),
        0.85,
      );
    }
  }
}

function applyHandDrivenPose(
  rig: AvatarRig | undefined,
  hands: TrackedHands,
): void {
  if (!trackingActive || !rig || (!hands.left && !hands.right)) return;
  for (const [side, hand] of Object.entries(hands) as Array<
    [HandSide, readonly ImagePoint[] | undefined]
  >) {
    if (!hand) continue;
    applyFingerPose(rig, side, hand);
  }
}

function prepareAvatar(avatar: THREE.Group): AvatarRig {
  const rig = prepareAvatarRig(avatar);
  avatar.position.copy(player.position);
  avatar.rotation.copy(player.rotation);
  scene.add(avatar);
  avatar.updateWorldMatrix(true, true);
  rig.bones = cacheRetargetBones(avatar);
  return rig;
}

async function loadPlayerAvatar(): Promise<void> {
  try {
    const avatar = await fbxLoader.loadAsync(
      "/assets/mixamo/character/y-bot.fbx",
    );
    scene.remove(player);
    playerAvatar = prepareAvatar(avatar);
  } catch (error) {
    console.error("Unable to load Y Bot", error);
  }
}

void loadPlayerAvatar();

window.addEventListener("keydown", (event) => {
  keys.add(event.key.toLowerCase());
  if (event.key.toLowerCase() === "r") {
    player.position.copy(PLAYER_START_POSITION);
    player.rotation.y = PLAYER_START_ROTATION_Y;
    playerStamina = 100;
  }
});
window.addEventListener("keyup", (event) =>
  keys.delete(event.key.toLowerCase()),
);

function setTrackingStatus(message: string, active = false): void {
  trackingStatus.textContent = message;
  trackingDot.classList.toggle("is-active", active);
}

const poseConnections: ReadonlyArray<readonly [number, number]> = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
];
const handConnections: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

function drawPoseOverlay(
  landmarks: readonly PosePoint[] | undefined,
  hands: readonly (readonly ImagePoint[])[],
): void {
  const width = webcam.clientWidth;
  const height = webcam.clientHeight;
  if (poseOverlay.width !== width || poseOverlay.height !== height) {
    poseOverlay.width = width;
    poseOverlay.height = height;
  }
  poseContext.clearRect(0, 0, width, height);
  if (!landmarks?.length || !webcam.videoWidth || !webcam.videoHeight) return;

  const videoScale = Math.max(
    width / webcam.videoWidth,
    height / webcam.videoHeight,
  );
  const renderedWidth = webcam.videoWidth * videoScale;
  const renderedHeight = webcam.videoHeight * videoScale;
  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;
  const pointFor = (landmark: PosePoint): [number, number] => [
    offsetX + landmark.x * renderedWidth,
    offsetY + landmark.y * renderedHeight,
  ];

  poseContext.lineCap = "round";
  poseContext.lineWidth = 2;
  poseContext.strokeStyle = "#42d8ae";
  for (const [firstIndex, secondIndex] of poseConnections) {
    const first = landmarks[firstIndex];
    const second = landmarks[secondIndex];
    if (!first || !second) continue;
    const [firstX, firstY] = pointFor(first);
    const [secondX, secondY] = pointFor(second);
    poseContext.beginPath();
    poseContext.moveTo(firstX, firstY);
    poseContext.lineTo(secondX, secondY);
    poseContext.stroke();
  }

  for (const index of new Set(poseConnections.flat())) {
    const landmark = landmarks[index];
    if (!landmark) continue;
    const [x, y] = pointFor(landmark);
    poseContext.fillStyle =
      ((landmark as typeof landmark & { visibility?: number }).visibility ??
      0 >= 0.45)
        ? "#42d8ae"
        : "#df9a4e";
    poseContext.beginPath();
    poseContext.arc(x, y, 3, 0, Math.PI * 2);
    poseContext.fill();
  }

  poseContext.strokeStyle = "#f2d08d";
  poseContext.lineWidth = 1.5;
  for (const hand of hands) {
    for (const [firstIndex, secondIndex] of handConnections) {
      const first = hand[firstIndex];
      const second = hand[secondIndex];
      if (!first || !second) continue;
      const [firstX, firstY] = pointFor(first);
      const [secondX, secondY] = pointFor(second);
      poseContext.beginPath();
      poseContext.moveTo(firstX, firstY);
      poseContext.lineTo(secondX, secondY);
      poseContext.stroke();
    }
    poseContext.fillStyle = "#f2d08d";
    for (const landmark of hand) {
      if (!landmark) continue;
      const [x, y] = pointFor(landmark);
      poseContext.beginPath();
      poseContext.arc(x, y, 1.75, 0, Math.PI * 2);
      poseContext.fill();
    }
  }
}

function assignHandsToSides(
  imageLandmarks: readonly ImagePoint[],
): TrackedHands {
  const assigned: TrackedHands = {};
  const unused: number[] = [];

  for (let index = 0; index < latestHandLandmarks.length; index += 1) {
    const label = latestHandLabels[index];
    const hand = latestHandLandmarks[index];
    if (!hand) continue;
    if (label === "Left" && !assigned.left) {
      assigned.left = hand;
      continue;
    }
    if (label === "Right" && !assigned.right) {
      assigned.right = hand;
      continue;
    }
    unused.push(index);
  }

  const nearestUnused = (wrist: ImagePoint): number =>
    unused.reduce((nearestIndex, index) => {
      if (nearestIndex === -1) return index;
      const hand = latestHandLandmarks[index];
      const nearest = latestHandLandmarks[nearestIndex];
      if (!hand?.[0] || !nearest?.[0]) return nearestIndex;
      const distance = Math.hypot(hand[0].x - wrist.x, hand[0].y - wrist.y);
      const nearestDistance = Math.hypot(
        nearest[0].x - wrist.x,
        nearest[0].y - wrist.y,
      );
      return distance < nearestDistance ? index : nearestIndex;
    }, -1);

  if (!assigned.left) {
    const leftIndex = nearestUnused(imageLandmarks[15]);
    if (leftIndex >= 0) {
      assigned.left = latestHandLandmarks[leftIndex];
      unused.splice(unused.indexOf(leftIndex), 1);
    }
  }
  if (!assigned.right) {
    const rightIndex = nearestUnused(imageLandmarks[16]);
    if (rightIndex >= 0) assigned.right = latestHandLandmarks[rightIndex];
  }

  return assigned;
}

async function startCamera(): Promise<void> {
  cameraButton.disabled = true;
  setTrackingStatus("carregando modelo...");

  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm",
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.5,
    });
    webcam.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    await webcam.play();
    trackingActive = true;
    cameraButton.textContent = "Camera ativa";
    setTrackingStatus("procurando postura...");
  } catch (error) {
    console.error("Unable to start camera pose tracking", error);
    cameraButton.disabled = false;
    setTrackingStatus("camera indisponivel");
  }
}

cameraButton.addEventListener("click", () => void startCamera());

function updateTrackedGloves(): void {
  if (
    !trackingActive ||
    !poseLandmarker ||
    webcam.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  )
    return;
  if (webcam.currentTime === lastVideoTime) return;

  lastVideoTime = webcam.currentTime;
  const timestamp = performance.now();
  const result = poseLandmarker.detectForVideo(webcam, timestamp);
  if (handLandmarker) {
    const handResult = handLandmarker.detectForVideo(webcam, timestamp);
    latestHandLandmarks =
      handResult.landmarks as readonly (readonly ImagePoint[])[];
    latestHandLabels = handResult.handednesses.map((categories) => {
      const name = categories[0]?.categoryName;
      return name === "Left" || name === "Right" ? name : undefined;
    });
  }
  const imageLandmarks = result.landmarks[0];
  const worldLandmarks = result.worldLandmarks[0];
  drawPoseOverlay(imageLandmarks, latestHandLandmarks);
  if (!imageLandmarks || !worldLandmarks) {
    setTrackingStatus("corpo fora do quadro");
    return;
  }

  const leftShoulder = worldLandmarks[11];
  const rightShoulder = worldLandmarks[12];
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);
  const poseLeftWrist = worldLandmarks[15];
  const poseRightWrist = worldLandmarks[16];
  const leftElbow = worldLandmarks[13];
  const rightElbow = worldLandmarks[14];
  const leftHip = worldLandmarks[23];
  const rightHip = worldLandmarks[24];
  const leftKnee = worldLandmarks[25];
  const rightKnee = worldLandmarks[26];
  const leftAnkle = worldLandmarks[27];
  const rightAnkle = worldLandmarks[28];
  latestTrackedHands = assignHandsToSides(imageLandmarks);
  const torsoVisibility = Math.min(
    imageLandmarks[11].visibility ?? 0,
    imageLandmarks[12].visibility ?? 0,
    imageLandmarks[23].visibility ?? 0,
    imageLandmarks[24].visibility ?? 0,
  );
  const wristVisibility = Math.min(
    imageLandmarks[15].visibility ?? 0,
    imageLandmarks[16].visibility ?? 0,
  );

  if (shoulderWidth < 0.08 || torsoVisibility < 0.45) {
    setTrackingStatus("mostre tronco e ombros");
    return;
  }
  if (wristVisibility < 0.45) {
    latestPose = undefined;
    setTrackingStatus("mostre as maos para os bracos");
    return;
  }

  latestPose = {
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    // Pose supplies the arm's full 3D target. Hand Landmarker remains local to palm orientation and fingers.
    leftWrist: poseLeftWrist,
    rightWrist: poseRightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
    shoulderWidth,
  };
  setTrackingStatus(
    latestHandLandmarks.length
      ? "corpo e maos ativos"
      : "rastreamento corporal ativo",
    true,
  );
}

function punch(fighter: THREE.Group, gloveName: string, amount: number): void {
  const glove = fighter.getObjectByName(gloveName)!;
  glove.position.z = THREE.MathUtils.damp(glove.position.z, amount, 18, 1 / 60);
}

function updateHud(): void {
  document.querySelector("#player-stamina")!.textContent =
    Math.round(playerStamina).toString();
  document.querySelector("#opponent-stamina")!.textContent =
    Math.round(opponentStamina).toString();
  (document.querySelector("#player-meter") as HTMLElement).style.width =
    `${playerStamina}%`;
  (document.querySelector("#opponent-meter") as HTMLElement).style.width =
    `${opponentStamina}%`;
}

function animate(time: number): void {
  const delta = Math.min((time - lastTime) / 1000, 0.05);
  lastTime = time;
  const moveSpeed = playerStamina < 25 ? 1.15 : 1.35;
  const movement = new THREE.Vector3(
    Number(keys.has("d")) - Number(keys.has("a")),
    0,
    Number(keys.has("s")) - Number(keys.has("w")),
  );
  if (movement.lengthSq()) {
    movement.normalize().multiplyScalar(moveSpeed * delta);
    player.position.add(movement);
    player.position.x = THREE.MathUtils.clamp(player.position.x, -3.65, 3.65);
    player.position.z = THREE.MathUtils.clamp(player.position.z, -3.65, 3.65);
    playerStamina = Math.max(0, playerStamina - 3 * delta);
  } else {
    playerStamina = Math.min(100, playerStamina + 12 * delta);
  }

  if (playerAvatar) {
    playerAvatar.avatar.position.copy(player.position);
    playerAvatar.avatar.rotation.copy(player.rotation);
  }

  const verticalAxis = new THREE.Vector3(0, 1, 0);
  const cameraOffset = new THREE.Vector3(-1.15, 2.05, -3.4)
    .applyAxisAngle(verticalAxis, player.rotation.y + cameraOrbitYaw)
    .applyAxisAngle(
      new THREE.Vector3(1, 0, 0).applyAxisAngle(
        verticalAxis,
        player.rotation.y + cameraOrbitYaw,
      ),
      cameraOrbitPitch,
    );
  const cameraTarget = player.position
    .clone()
    .add(
      new THREE.Vector3(0, 1.25, 0.35).applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        player.rotation.y,
      ),
    );
  camera.position.lerp(
    player.position.clone().add(cameraOffset),
    1 - Math.exp(-7 * delta),
  );
  camera.lookAt(cameraTarget);

  updateTrackedGloves();
  if (trackingActive && latestPose && playerAvatar) {
    applyUpperBodyPose(playerAvatar, latestPose);
  } else if (playerAvatar) {
    applyGuardPose(playerAvatar);
    punch(
      player,
      "left-glove",
      keys.has("j") && playerStamina >= 5 ? 0.9 : 0.08,
    );
    punch(
      player,
      "right-glove",
      keys.has("k") && playerStamina >= 7 ? 1.05 : 0.08,
    );
  }
  if (trackingActive) applyHandDrivenPose(playerAvatar, latestTrackedHands);
  updateHud();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

function resize(): void {
  const { width, height } = canvas.getBoundingClientRect();
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(canvas);
resize();
requestAnimationFrame(animate);
