import * as THREE from 'three'
import type { Joint, JointType } from './Joint'
import { clamp, swingTwistDecompose } from '../utils/math'

/**
 * 4 types, ball, hinge, free, fixed
 */
export interface HingeLimits {
  axis: THREE.Vector3       // local hinge axis (normalised)
  min: number               // radians
  max: number               // radians
}
export interface BallLimits {
  swingConeAngle: number    // max radians off the rest axis
}

const HINGE_DEFAULTS: HingeLimits = {
  axis: new THREE.Vector3(1, 0, 0),
  min: 0,
  max: Math.PI * 0.95,
}

const BALL_DEFAULTS: BallLimits = {
  swingConeAngle: Math.PI / 2,  // 90°
}

const HINGE_OVERRIDES: Record<number, Partial<HingeLimits>> = {
  // knee joints (25, 26, 27, 28) — only flex forward (0 → 150°)
  25: { min: 0,          max: Math.PI * 0.833 },
  26: { min: 0,          max: Math.PI * 0.833 },
  27: { min: 0,          max: Math.PI * 0.833 },
  28: { min: 0,          max: Math.PI * 0.833 },
  // elbow joints (13, 14, 15, 16) — flex 0 → 145°
  13: { min: 0,          max: Math.PI * 0.806 },
  14: { min: 0,          max: Math.PI * 0.806 },
  15: { min: 0,          max: Math.PI * 0.806 },
  16: { min: 0,          max: Math.PI * 0.806 },
}

export function applyConstraint(joint:Joint):void{
    const type: JointType = joint.jointType
    if (type === 'fixed') {
        joint.localRotation.set(0, 0, 0)
        return
    }
    if (type === 'free') {
        // No limits — leave rotation as-is.
        return
    }

    if (type === 'hinge'){
        const defaults = HINGE_DEFAULTS
        const overrides = HINGE_OVERRIDES[joint.id] ?? {}
        const limits: HingeLimits = { ...defaults, ...overrides }

        // Project the full rotation onto the hinge axis and clamp.
        const q = new THREE.Quaternion().setFromEuler(joint.localRotation)
        const axis = limits.axis.clone().normalize()

        const hingeQ  = swingTwistDecompose(q, axis)[1]

        let angle = 2 * Math.acos(clamp(hingeQ.w, -1, 1))

        if (hingeQ.x * axis.x + hingeQ.y * axis.y + hingeQ.z * axis.z < 0) angle = -angle
        angle = clamp(angle, limits.min, limits.max)
    
        const clamped = new THREE.Quaternion().setFromAxisAngle(axis, angle)
        joint.localRotation.setFromQuaternion(clamped)
        return
    }

    if (type === 'ball') {
        const limits: BallLimits = BALL_DEFAULTS
        const q = new THREE.Quaternion().setFromEuler(joint.localRotation)

        const restAxis = joint.restOffset.clone().normalize()
        const [swing, twist] = swingTwistDecompose(q, restAxis)

        const swingAngle = 2 * Math.acos(clamp(swing.w, -1, 1))

        if (swingAngle > limits.swingConeAngle){
            const swingAxis = new THREE.Vector3(swing.x, swing.y, swing.z)
            if (swingAxis.lengthSq() > 1e-10){
                swingAxis.normalize()
                const limitedSwing = new THREE.Quaternion().setFromAxisAngle(swingAxis, limits.swingConeAngle)
                const clamped = limitedSwing.multiply(twist)
                joint.localRotation.setFromQuaternion(clamped)
            }
        }
    }
}