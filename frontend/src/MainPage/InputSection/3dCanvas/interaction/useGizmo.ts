import { useState, useCallback } from 'react'
import * as THREE from 'three'
import type { Joint } from '../skeleton/Joint'
import { applyConstraint } from '../skeleton/constraints'

export type GizmoAxis = 'x' | 'y' | 'z' | null

export interface UseGizmoReturn {
  activeAxis: GizmoAxis
  setActiveAxis: (axis: GizmoAxis) => void
  /** Apply a delta rotation (radians) around the active axis to the joint. */
  applyDelta: (joint: Joint, deltaAngle: number, onUpdate: () => void) => void
  /** Release the gizmo (mouse up). */
  release: () => void
}

export function useGizmo(): UseGizmoReturn {
  const [activeAxis, setActiveAxis] = useState<GizmoAxis>(null)
 
  const applyDelta = useCallback(
    (joint: Joint, deltaAngle: number, onUpdate: () => void) => {
      if (!activeAxis) return
 
      // Build a delta quaternion around the chosen local axis.
      const axis = new THREE.Vector3(
        activeAxis === 'x' ? 1 : 0,
        activeAxis === 'y' ? 1 : 0,
        activeAxis === 'z' ? 1 : 0,
      )
      const deltaQ = new THREE.Quaternion().setFromAxisAngle(axis, deltaAngle)
 
      // Compose with current local rotation.
      const currentQ = new THREE.Quaternion().setFromEuler(joint.localRotation)
      currentQ.multiply(deltaQ)
      joint.localRotation.setFromQuaternion(currentQ)
 
      // Clamp immediately.
      applyConstraint(joint)
 
      // Notify parent to re-run FK and re-render.
      onUpdate()
    },
    [activeAxis],
  )
 
  const release = useCallback(() => {
    setActiveAxis(null)
  }, [])
 
  return { activeAxis, setActiveAxis, applyDelta, release }
}