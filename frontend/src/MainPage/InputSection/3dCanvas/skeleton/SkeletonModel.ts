import * as THREE from 'three'
import type { Joint } from './Joint'
import type { Bone } from './Bone'
import { updateSkeletonFK, getRoots } from './kinematics'
import { JOINT_TYPES } from '../data'

export class SkeletonModel {
  joints: Joint[]
  bones: Bone[]

  constructor(embedding: [number, number, number][], edges: [number, number][]) {
    this.joints = embedding.map((p, i) => ({
      id: i,
      position: p,
      parent: null,
      children: [],
      localRotation: new THREE.Euler(0, 0, 0, 'XYZ'),
      restOffset: new THREE.Vector3(0, 0, 0),
      boneLength: 0,   
      jointType: JOINT_TYPES[i] ?? 'free',
    }))

    this.bones = []

    edges.forEach(([a, b]) => {
      if (a >= this.joints.length || b >= this.joints.length) return

      const parent  = this.joints[a]
      const child   = this.joints[b]

      if (!child.parent) {
        child.parent = parent
        parent.children.push(child)
        const offset = new THREE.Vector3(
          child.position[0] - parent.position[0],
          child.position[1] - parent.position[1],
          child.position[2] - parent.position[2],
        )
        child.restOffset = offset.clone()
        child.boneLength = offset.length()
      }

      const start = parent
      const end   = child
      const dx = end.position[0] - start.position[0]
      const dy = end.position[1] - start.position[1]
      const dz = end.position[2] - start.position[2]
      const length = Math.sqrt(dx * dx + dy * dy + dz * dz)

      this.bones.push({ start, end, length })
    })

    this._runFK()

  }

  /**
   * After mutating any joint's `localRotation`, call this to recompute
   * all world-space positions via forward kinematics.
   */
  applyRotations(): void {
    this._runFK()
  }

  /**
   * Set one joint's local rotation (Euler XYZ radians) and re-run FK.
   */
  setJointRotation(jointId: number, x: number, y: number, z: number): void {
    const joint = this.joints[jointId]
    if (!joint) return
    joint.localRotation.set(x, y, z)
    this._runFK()
  }

   /**
   * Reset all rotations to zero (bind pose) and re-run FK.
   */
  resetToBindPose(): void {
    for (const j of this.joints) {
      j.localRotation.set(0, 0, 0)
    }
    this._runFK()
  }


  private _runFK(): void {
    updateSkeletonFK(getRoots(this.joints))
  }
}