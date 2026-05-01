import { Canvas } from '@react-three/fiber'
import { forwardRef, useCallback, useMemo, useState } from 'react'
import { SkeletonModel } from '../skeleton/SkeletonModel'
import type { SkeletonCanvasRef } from '../../types'
import { edges, embedding } from '../data'
import { useGizmo } from '../interaction/useGizmo'
import { AxisPicker } from '../interaction/AxisPicker'
import { Scene } from './SkeletonScene'
import { css, classNames } from './SkeletonCanvas.styles'

// Scene has been moved to SkeletonScene.tsx and is imported above.

export const SkeletonCanvas = forwardRef<SkeletonCanvasRef>((props, ref) => {
  const model = useMemo(() => {
    // Center + scale the embedding.
    const raw = embedding
    const cx = raw.reduce((s, p) => s + p[0], 0) / raw.length
    const cy = raw.reduce((s, p) => s + p[1], 0) / raw.length
    const cz = raw.reduce((s, p) => s + p[2], 0) / raw.length
    const scale = 3
    const centered: [number, number, number][] = raw.map(p => [
      (p[0] - cx) * scale,
      (p[1] - cy) * scale,
      (p[2] - cz) * scale,
    ])
    return new SkeletonModel(centered, edges)
  }, [])

  const [, setTick] = useState(0)
  const forceUpdate = useCallback(() => setTick(t => t + 1), [])
  
  const gizmo = useGizmo()


  return (
    <div className={classNames.root}>
      <style>{css}</style>
      <Canvas camera={{ position: [0, 0, 3.5], fov: 55 }}>
        <Scene model={model} ref={ref} onModelUpdate={forceUpdate} gizmo={gizmo} />
      </Canvas>

      {/* Rotation axis picker overlay */}
      <div className={classNames.overlay}>
        <div className={classNames.overlayText}>
          {gizmo.activeAxis
            ? `Rotate ${gizmo.activeAxis.toUpperCase()} axis — drag a joint`
            : 'Select an axis then drag a joint'}
        </div>
        <AxisPicker activeAxis={gizmo.activeAxis} setActiveAxis={gizmo.setActiveAxis} />
        <button
          onClick={() => { model.resetToBindPose(); forceUpdate() }}
          className={classNames.button}
        >
          Reset pose
        </button>
      </div>
    </div>
  )
})