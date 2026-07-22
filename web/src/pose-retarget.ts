import * as THREE from "three";

export type PosePoint = { x: number; y: number; z: number };

export type TrackedPose = {
  leftShoulder: PosePoint;
  rightShoulder: PosePoint;
  leftElbow: PosePoint;
  rightElbow: PosePoint;
  leftWrist: PosePoint;
  rightWrist: PosePoint;
  leftHip: PosePoint;
  rightHip: PosePoint;
  leftKnee: PosePoint;
  rightKnee: PosePoint;
  leftAnkle: PosePoint;
  rightAnkle: PosePoint;
  shoulderWidth: number;
};

export type BoneRestPose = {
  bone: THREE.Bone;
  childBone: THREE.Bone;
  restLocalQuaternion: THREE.Quaternion;
  restQuaternionInRoot: THREE.Quaternion;
  restDirectionInRoot: THREE.Vector3;
  restPalmNormalInRoot: THREE.Vector3;
};

export type AvatarRig = {
  avatar: THREE.Group;
  bones: Map<string, BoneRestPose>;
};

function firstChildBone(bone: THREE.Bone): THREE.Bone | undefined {
  return bone.children.find(
    (child): child is THREE.Bone =>
      child instanceof THREE.Bone && child.name !== bone.name,
  );
}

function preferredChildBone(
  bone: THREE.Bone,
  boneName: string,
): THREE.Bone | undefined {
  if (boneName.endsWith("Hand")) {
    const side = boneName.includes("Left") ? "Left" : "Right";
    const middle = bone.getObjectByName(`mixamorig${side}HandMiddle1`);
    if (middle instanceof THREE.Bone) return middle;
  }
  return firstChildBone(bone);
}

export function cacheRetargetBones(
  avatar: THREE.Group,
): Map<string, BoneRestPose> {
  const retargetBones = new Map<string, BoneRestPose>();
  const handBoneNames = ["Left", "Right"].flatMap((side) => [
    `mixamorig${side}Hand`,
    ...["Thumb", "Index", "Middle", "Ring", "Pinky"].flatMap((finger) =>
      [1, 2, 3].map((segment) => `mixamorig${side}Hand${finger}${segment}`),
    ),
  ]);
  const boneNames = [
    "mixamorigHips",
    "mixamorigSpine", "mixamorigSpine1", "mixamorigSpine2", "mixamorigNeck",
    "mixamorigLeftArm",
    "mixamorigLeftForeArm",
    "mixamorigRightArm",
    "mixamorigRightForeArm",
    "mixamorigLeftUpLeg", "mixamorigLeftLeg",
    "mixamorigRightUpLeg", "mixamorigRightLeg",
    ...handBoneNames,
  ];
  const rootWorldQuaternion = avatar.getWorldQuaternion(new THREE.Quaternion());
  const inverseRootQuaternion = rootWorldQuaternion.clone().invert();

  for (const name of boneNames) {
    const bone = avatar.getObjectByName(name);
    if (!(bone instanceof THREE.Bone)) continue;
    const childBone = preferredChildBone(bone, name);
    if (!childBone) continue;

    const bonePosition = bone.getWorldPosition(new THREE.Vector3());
    const childPosition = childBone.getWorldPosition(new THREE.Vector3());
    const restDirection = childPosition.sub(bonePosition).normalize();
    let restPalmNormalInRoot = new THREE.Vector3(0, -1, 0);
    if (name.endsWith("Hand")) {
      const side = name.includes("Left") ? "Left" : "Right";
      const index = avatar.getObjectByName(`mixamorig${side}HandIndex1`);
      const pinky = avatar.getObjectByName(`mixamorig${side}HandPinky1`);
      if (index instanceof THREE.Bone && pinky instanceof THREE.Bone) {
        const indexPosition = index.getWorldPosition(new THREE.Vector3());
        const pinkyPosition = pinky.getWorldPosition(new THREE.Vector3());
        const acrossPalm = indexPosition.sub(pinkyPosition).normalize();
        const restPalmNormalWorld = new THREE.Vector3()
          .crossVectors(restDirection, acrossPalm)
          .normalize();
        if (restPalmNormalWorld.lengthSq() > 0.0001) {
          restPalmNormalInRoot = restPalmNormalWorld.applyQuaternion(
            inverseRootQuaternion,
          );
        }
      }
    }
    retargetBones.set(name, {
      bone,
      childBone,
      restLocalQuaternion: bone.quaternion.clone(),
      restQuaternionInRoot: inverseRootQuaternion
        .clone()
        .multiply(bone.getWorldQuaternion(new THREE.Quaternion())),
      restDirectionInRoot: restDirection.applyQuaternion(inverseRootQuaternion),
      restPalmNormalInRoot,
    });
  }
  return retargetBones;
}

export function rotateBoneToward(
  rig: AvatarRig,
  name: string,
  directionInRoot: THREE.Vector3,
  strength: number,
): void {
  const restPose = rig.bones.get(name);
  if (!restPose || directionInRoot.lengthSq() < 0.0001) return;

  const rootWorldQuaternion = rig.avatar.getWorldQuaternion(
    new THREE.Quaternion(),
  );
  const restDirectionWorld = restPose.restDirectionInRoot
    .clone()
    .applyQuaternion(rootWorldQuaternion)
    .normalize();
  const targetDirectionWorld = directionInRoot
    .clone()
    .applyQuaternion(rootWorldQuaternion)
    .normalize();
  const correction = new THREE.Quaternion().setFromUnitVectors(
    restDirectionWorld,
    targetDirectionWorld,
  );
  const restWorldQuaternion = rootWorldQuaternion
    .clone()
    .multiply(restPose.restQuaternionInRoot);
  const targetWorldQuaternion = correction.multiply(restWorldQuaternion);
  const blendedWorldQuaternion = restWorldQuaternion.slerp(
    targetWorldQuaternion,
    strength,
  );
  const parentWorldQuaternion = restPose.bone.parent!.getWorldQuaternion(
    new THREE.Quaternion(),
  );

  restPose.bone.quaternion.copy(
    parentWorldQuaternion.invert().multiply(blendedWorldQuaternion),
  );
  restPose.bone.updateWorldMatrix(false, true);
}

export function orientBoneWithPalm(
  rig: AvatarRig,
  name: string,
  forwardInRoot: THREE.Vector3,
  palmNormalInRoot: THREE.Vector3,
  strength: number,
): void {
  const restPose = rig.bones.get(name);
  if (!restPose || forwardInRoot.lengthSq() < 0.0001) return;

  const rootWorldQuaternion = rig.avatar.getWorldQuaternion(
    new THREE.Quaternion(),
  );
  const restDirectionWorld = restPose.restDirectionInRoot
    .clone()
    .applyQuaternion(rootWorldQuaternion)
    .normalize();
  const targetDirectionWorld = forwardInRoot
    .clone()
    .normalize()
    .applyQuaternion(rootWorldQuaternion)
    .normalize();
  const align = new THREE.Quaternion().setFromUnitVectors(
    restDirectionWorld,
    targetDirectionWorld,
  );
  const restWorldQuaternion = rootWorldQuaternion
    .clone()
    .multiply(restPose.restQuaternionInRoot);
  let targetWorldQuaternion = align.clone().multiply(restWorldQuaternion);

  const alignedPalmWorld = restPose.restPalmNormalInRoot
    .clone()
    .applyQuaternion(rootWorldQuaternion)
    .applyQuaternion(align)
    .normalize();
  const desiredPalmWorld = palmNormalInRoot
    .clone()
    .normalize()
    .applyQuaternion(rootWorldQuaternion)
    .normalize();
  const alignedPlane = alignedPalmWorld.addScaledVector(
    targetDirectionWorld,
    -alignedPalmWorld.dot(targetDirectionWorld),
  );
  const desiredPlane = desiredPalmWorld.addScaledVector(
    targetDirectionWorld,
    -desiredPalmWorld.dot(targetDirectionWorld),
  );
  if (alignedPlane.lengthSq() > 0.0001 && desiredPlane.lengthSq() > 0.0001) {
    const twist = new THREE.Quaternion().setFromUnitVectors(
      alignedPlane.normalize(),
      desiredPlane.normalize(),
    );
    targetWorldQuaternion = twist.multiply(targetWorldQuaternion);
  }

  const blendedWorldQuaternion = restWorldQuaternion
    .clone()
    .slerp(targetWorldQuaternion, strength);
  const parentWorldQuaternion = restPose.bone.parent!.getWorldQuaternion(
    new THREE.Quaternion(),
  );
  restPose.bone.quaternion.copy(
    parentWorldQuaternion.invert().multiply(blendedWorldQuaternion),
  );
  restPose.bone.updateWorldMatrix(false, true);
}

function solveArm(
  rig: AvatarRig,
  upperArmName: string,
  foreArmName: string,
  shoulder: THREE.Vector3,
  elbow: THREE.Vector3,
  wrist: THREE.Vector3,
): void {
  const upperArm = rig.bones.get(upperArmName);
  const foreArm = rig.bones.get(foreArmName);
  if (!upperArm || !foreArm) return;

  const shoulderWorld = upperArm.bone.getWorldPosition(new THREE.Vector3());
  const elbowWorld = foreArm.bone.getWorldPosition(new THREE.Vector3());
  const wristWorld = foreArm.childBone.getWorldPosition(new THREE.Vector3());
  const upperArmLength = shoulderWorld.distanceTo(elbowWorld);
  const foreArmLength = elbowWorld.distanceTo(wristWorld);
  const sourceUpperArm = elbow.clone().sub(shoulder);
  const sourceWrist = wrist.clone().sub(shoulder);
  const oppositeUpperArmName = upperArmName.includes('Left')
    ? 'mixamorigRightArm'
    : 'mixamorigLeftArm';
  const oppositeUpperArm = rig.bones.get(oppositeUpperArmName);
  const avatarShoulderWidth = oppositeUpperArm
    ? shoulderWorld.distanceTo(oppositeUpperArm.bone.getWorldPosition(new THREE.Vector3()))
    : 0;
  if (upperArmLength < 0.001 || foreArmLength < 0.001 || avatarShoulderWidth < 0.001 || sourceWrist.lengthSq() < 0.0001)
    return;

  const reach = upperArmLength + foreArmLength;
  const minimumReach = Math.abs(upperArmLength - foreArmLength) + 0.001;
  const targetDistance = THREE.MathUtils.clamp(
    sourceWrist.length() * avatarShoulderWidth,
    minimumReach,
    reach - 0.001,
  );
  const targetDirection = sourceWrist.normalize();
  const targetWorld = shoulderWorld
    .clone()
    .addScaledVector(
      targetDirection.applyQuaternion(
        rig.avatar.getWorldQuaternion(new THREE.Quaternion()),
      ),
      targetDistance,
    );
  const shoulderToTarget = targetWorld.clone().sub(shoulderWorld);
  const shoulderToTargetLength = shoulderToTarget.length();
  if (shoulderToTargetLength < 0.001) return;

  const axis = shoulderToTarget.normalize();
  const sourceElbowDirection = sourceUpperArm
    .normalize()
    .applyQuaternion(rig.avatar.getWorldQuaternion(new THREE.Quaternion()));
  const elbowPlane = sourceElbowDirection.addScaledVector(
    axis,
    -sourceElbowDirection.dot(axis),
  );
  if (elbowPlane.lengthSq() < 0.0001)
    elbowPlane.copy(new THREE.Vector3(0, 1, 0).cross(axis));
  elbowPlane.normalize();

  const elbowAlongAxis =
    (upperArmLength ** 2 - foreArmLength ** 2 + shoulderToTargetLength ** 2) /
    (2 * shoulderToTargetLength);
  const elbowOffset = Math.sqrt(
    Math.max(0, upperArmLength ** 2 - elbowAlongAxis ** 2),
  );
  const solvedElbowWorld = shoulderWorld
    .clone()
    .addScaledVector(axis, elbowAlongAxis)
    .addScaledVector(elbowPlane, elbowOffset);
  const inverseRootQuaternion = rig.avatar
    .getWorldQuaternion(new THREE.Quaternion())
    .invert();

  rotateBoneToward(
    rig,
    upperArmName,
    solvedElbowWorld.sub(shoulderWorld).applyQuaternion(inverseRootQuaternion),
    1,
  );
  rotateBoneToward(
    rig,
    foreArmName,
    targetWorld
      .sub(foreArm.bone.getWorldPosition(new THREE.Vector3()))
      .applyQuaternion(inverseRootQuaternion),
    1,
  );
}

function solveLeg(
  rig: AvatarRig,
  upperLegName: string,
  lowerLegName: string,
  hip: THREE.Vector3,
  knee: THREE.Vector3,
  ankle: THREE.Vector3,
): void {
  const upperLeg = rig.bones.get(upperLegName);
  const lowerLeg = rig.bones.get(lowerLegName);
  const leftArm = rig.bones.get('mixamorigLeftArm');
  const rightArm = rig.bones.get('mixamorigRightArm');
  if (!upperLeg || !lowerLeg || !leftArm || !rightArm) return;

  const hipWorld = upperLeg.bone.getWorldPosition(new THREE.Vector3());
  const kneeWorld = lowerLeg.bone.getWorldPosition(new THREE.Vector3());
  const ankleWorld = lowerLeg.childBone.getWorldPosition(new THREE.Vector3());
  const upperLegLength = hipWorld.distanceTo(kneeWorld);
  const lowerLegLength = kneeWorld.distanceTo(ankleWorld);
  const sourceUpperLeg = knee.clone().sub(hip);
  const sourceAnkle = ankle.clone().sub(hip);
  const avatarShoulderWidth = leftArm.bone.getWorldPosition(new THREE.Vector3())
    .distanceTo(rightArm.bone.getWorldPosition(new THREE.Vector3()));
  if (upperLegLength < 0.001 || lowerLegLength < 0.001 || avatarShoulderWidth < 0.001 || sourceAnkle.lengthSq() < 0.0001) return;

  const reach = upperLegLength + lowerLegLength;
  const minimumReach = Math.abs(upperLegLength - lowerLegLength) + 0.001;
  const targetDistance = THREE.MathUtils.clamp(
    sourceAnkle.length() * avatarShoulderWidth,
    minimumReach,
    reach - 0.001,
  );
  const targetWorld = hipWorld.clone().addScaledVector(
    sourceAnkle.normalize().applyQuaternion(rig.avatar.getWorldQuaternion(new THREE.Quaternion())),
    targetDistance,
  );
  const hipToTarget = targetWorld.clone().sub(hipWorld);
  const hipToTargetLength = hipToTarget.length();
  if (hipToTargetLength < 0.001) return;

  const axis = hipToTarget.normalize();
  const sourceKneeDirection = sourceUpperLeg.normalize().applyQuaternion(rig.avatar.getWorldQuaternion(new THREE.Quaternion()));
  const kneePlane = sourceKneeDirection.addScaledVector(axis, -sourceKneeDirection.dot(axis));
  if (kneePlane.lengthSq() < 0.0001) kneePlane.copy(new THREE.Vector3(0, 0, 1).cross(axis));
  kneePlane.normalize();
  const kneeAlongAxis = (upperLegLength ** 2 - lowerLegLength ** 2 + hipToTargetLength ** 2) / (2 * hipToTargetLength);
  const kneeOffset = Math.sqrt(Math.max(0, upperLegLength ** 2 - kneeAlongAxis ** 2));
  const solvedKneeWorld = hipWorld.clone()
    .addScaledVector(axis, kneeAlongAxis)
    .addScaledVector(kneePlane, kneeOffset);
  const inverseRootQuaternion = rig.avatar.getWorldQuaternion(new THREE.Quaternion()).invert();

  rotateBoneToward(rig, upperLegName, solvedKneeWorld.sub(hipWorld).applyQuaternion(inverseRootQuaternion), 1);
  rotateBoneToward(
    rig,
    lowerLegName,
    targetWorld.sub(lowerLeg.bone.getWorldPosition(new THREE.Vector3())).applyQuaternion(inverseRootQuaternion),
    1,
  );
}

function poseToAvatarSpace(
  point: PosePoint,
  shoulderCenter: PosePoint,
  shoulderWidth: number,
): THREE.Vector3 {
  return new THREE.Vector3(
    (point.x - shoulderCenter.x) / shoulderWidth,
    (shoulderCenter.y - point.y) / shoulderWidth,
    (shoulderCenter.z - point.z) / shoulderWidth,
  );
}

export function applyUpperBodyPose(rig: AvatarRig, pose: TrackedPose): void {
  if (rig.bones.size === 0) return;

  for (const restPose of rig.bones.values()) {
    restPose.bone.quaternion.copy(restPose.restLocalQuaternion);
  }
  rig.avatar.updateWorldMatrix(true, true);

  const {
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
    shoulderWidth,
  } = pose;
  const shoulderCenter = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
    z: (leftShoulder.z + rightShoulder.z) / 2,
  };
  const hipCenter = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
    z: (leftHip.z + rightHip.z) / 2,
  };
  const leftShoulderPoint = poseToAvatarSpace(
    leftShoulder,
    shoulderCenter,
    shoulderWidth,
  );
  const rightShoulderPoint = poseToAvatarSpace(
    rightShoulder,
    shoulderCenter,
    shoulderWidth,
  );
  const leftElbowPoint = poseToAvatarSpace(
    leftElbow,
    shoulderCenter,
    shoulderWidth,
  );
  const rightElbowPoint = poseToAvatarSpace(
    rightElbow,
    shoulderCenter,
    shoulderWidth,
  );
  const leftWristPoint = poseToAvatarSpace(
    leftWrist,
    shoulderCenter,
    shoulderWidth,
  );
  const rightWristPoint = poseToAvatarSpace(
    rightWrist,
    shoulderCenter,
    shoulderWidth,
  );
  const hipCenterPoint = poseToAvatarSpace(
    hipCenter,
    shoulderCenter,
    shoulderWidth,
  );
  const leftHipPoint = poseToAvatarSpace(leftHip, shoulderCenter, shoulderWidth);
  const rightHipPoint = poseToAvatarSpace(rightHip, shoulderCenter, shoulderWidth);
  const leftKneePoint = poseToAvatarSpace(leftKnee, shoulderCenter, shoulderWidth);
  const rightKneePoint = poseToAvatarSpace(rightKnee, shoulderCenter, shoulderWidth);
  const leftAnklePoint = poseToAvatarSpace(leftAnkle, shoulderCenter, shoulderWidth);
  const rightAnklePoint = poseToAvatarSpace(rightAnkle, shoulderCenter, shoulderWidth);

  const torsoDirection = hipCenterPoint.clone().negate();
  torsoDirection.x *= 1.8;
  torsoDirection.z *= 1.8;
  rotateBoneToward(rig, "mixamorigSpine", torsoDirection, 0.72);
  rotateBoneToward(rig, "mixamorigSpine1", torsoDirection, 0.56);
  rotateBoneToward(rig, "mixamorigSpine2", torsoDirection, 0.38);
  solveArm(
    rig,
    "mixamorigLeftArm",
    "mixamorigLeftForeArm",
    leftShoulderPoint,
    leftElbowPoint,
    leftWristPoint,
  );
  solveArm(
    rig,
    "mixamorigRightArm",
    "mixamorigRightForeArm",
    rightShoulderPoint,
    rightElbowPoint,
    rightWristPoint,
  );
  solveLeg(rig, "mixamorigLeftUpLeg", "mixamorigLeftLeg", leftHipPoint, leftKneePoint, leftAnklePoint);
  solveLeg(rig, "mixamorigRightUpLeg", "mixamorigRightLeg", rightHipPoint, rightKneePoint, rightAnklePoint);
}

export function applyGuardPose(rig: AvatarRig): void {
  if (rig.bones.size === 0) return;

  for (const restPose of rig.bones.values()) {
    restPose.bone.quaternion.copy(restPose.restLocalQuaternion);
  }
  rig.avatar.updateWorldMatrix(true, true);

  rotateBoneToward(
    rig,
    "mixamorigLeftArm",
    new THREE.Vector3(-0.5, 0.05, 0.7),
    1,
  );
  rotateBoneToward(
    rig,
    "mixamorigLeftForeArm",
    new THREE.Vector3(0, 0.6, 0.55),
    1,
  );
  rotateBoneToward(
    rig,
    "mixamorigRightArm",
    new THREE.Vector3(0.5, 0.05, 0.7),
    1,
  );
  rotateBoneToward(
    rig,
    "mixamorigRightForeArm",
    new THREE.Vector3(0, 0.6, 0.55),
    1,
  );
}

export function prepareAvatarRig(avatar: THREE.Group, scale = 0.01): AvatarRig {
  avatar.scale.setScalar(scale);
  avatar.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  avatar.updateWorldMatrix(true, true);
  return { avatar, bones: cacheRetargetBones(avatar) };
}
