import { useRef, useCallback } from 'react'
import type { Joint } from '../skeleton/Joint'
import type { UseGizmoReturn } from './useGizmo'

interface UseDragOptions {
  gizmo: UseGizmoReturn
  /** Called after every rotation update so the parent can re-run FK. */
  onUpdate: (joint: Joint) => void
  /** Sensitivity: radians per pixel of drag. Default 0.01. */
  sensitivity?: number
}

export interface UseDragReturn {
  /** Attach to the joint mesh's onPointerDown event. */
  onPointerDown: (joint: Joint, event: PointerEvent) => void
}

export function useDrag({ gizmo, onUpdate, sensitivity = 0.01 }: UseDragOptions): UseDragReturn {
  const draggingJoint = useRef<Joint | null>(null)
  const lastPointer = useRef<{ x: number; y: number } | null>(null)
 
  const onPointerMove = useCallback(
    (e: PointerEvent) => {
      if (!draggingJoint.current || !lastPointer.current || !gizmo.activeAxis) return
 
      const dx = e.clientX - lastPointer.current.x
      const dy = e.clientY - lastPointer.current.y
      lastPointer.current = { x: e.clientX, y: e.clientY }
 
      // Choose drag axis to angle mapping.
      const delta = gizmo.activeAxis === 'x' ? -dy * sensitivity : dx * sensitivity
 
      gizmo.applyDelta(draggingJoint.current, delta, () =>
        onUpdate(draggingJoint.current!),
      )
    },
    [gizmo, onUpdate, sensitivity],
  )
 
  const onPointerUp = useCallback(
    (e: PointerEvent) => {
      draggingJoint.current = null
      lastPointer.current = null
      gizmo.release()
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
    },
    [gizmo, onPointerMove],
  )
 
  const onPointerDown = useCallback(
    (joint: Joint, e: PointerEvent) => {
      if (!gizmo.activeAxis) return  // Only drag when an axis is selected.
      e.stopPropagation()
      draggingJoint.current = joint
      lastPointer.current = { x: e.clientX, y: e.clientY }
      window.addEventListener('pointermove', onPointerMove)
      window.addEventListener('pointerup', onPointerUp)
    },
    [gizmo.activeAxis, onPointerMove, onPointerUp],
  )
 
  return { onPointerDown }
}