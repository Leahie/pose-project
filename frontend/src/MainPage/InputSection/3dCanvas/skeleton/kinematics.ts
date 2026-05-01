import * as THREE from 'three'
import type { Joint } from './Joint'

export function propagateFK(
    joint: Joint,
    parentWorldPos: THREE.Vector3 = new THREE.Vector3(), 
    parentWorldRot: THREE.Quaternion = new THREE.Quaternion(),
): void {
    // constrain it 
    const localQ = new THREE.Quaternion().setFromEuler(joint.localRotation)
    const worldRot = parentWorldRot.clone().multiply(localQ) 

    const rotatedOffset = joint.restOffset.clone().applyQuaternion(parentWorldRot)
    const worldPos = parentWorldPos.clone().add(rotatedOffset)

    joint.position = [worldPos.x, worldPos.y, worldPos.z]

    for (const child of joint.children) {
        propagateFK(child, worldPos, worldRot)
    }

}

export function updateSkeletonFK(roots: Joint[]):void {
    for (const root of roots){
        propagateFK(root, new THREE.Vector3(...root.position), new THREE.Quaternion())
    }
}

export function getRoots(joints: Joint[]): Joint[]{
    return joints.filter(j => j.parent == null)
}

export function rotateJoint(
    joint: Joint, 
    euler: THREE.Euler, 
    parentWorldPos: THREE.Vector3, 
    parentWorldRot: THREE.Quaternion,
):void{
    joint.localRotation.copy(euler)
    propagateFK(joint, parentWorldPos, parentWorldRot)
}