import { useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { forwardRef, useCallback, useImperativeHandle, useRef } from 'react'
import { JointMesh } from './JointMesh'
import { BoneLine } from './BoneLine'
import type { SkeletonCanvasRef } from '../../types'
import type { SkeletonModel } from '../skeleton/SkeletonModel'
import * as THREE from 'three'
import { useSelection } from '../interaction/useSelection'
import type { UseGizmoReturn } from '../interaction/useGizmo'
import type { Joint } from '../skeleton/Joint'
import { useDrag } from '../interaction/useDrag'

interface SceneProps {
  model: SkeletonModel;
  onModelUpdate: () => void
  gizmo: UseGizmoReturn
}

export const Scene = forwardRef<SkeletonCanvasRef, SceneProps>(({model, onModelUpdate, gizmo}, ref) => {
  const { camera } = useThree()
  const orbitRef = useRef<any>(null)

  const { selectedJoint, selectJoint } = useSelection()

  const handleJointUpdate = useCallback((joint: Joint) => {
    model.applyRotations()
    onModelUpdate()
  }, [model, onModelUpdate])

  const drag = useDrag({ gizmo, onUpdate: handleJointUpdate })

  const handleJointPointerDown = useCallback((joint: Joint, e: PointerEvent) => {
    if (gizmo.activeAxis) {
      if (orbitRef.current) orbitRef.current.enabled = false
      drag.onPointerDown(joint, e)
      const cleanup = () => {
        if (orbitRef.current) orbitRef.current.enabled = true
        window.removeEventListener('pointerup', cleanup)
      }
      window.addEventListener('pointerup', cleanup)
    }
  }, [gizmo.activeAxis, drag])

  useImperativeHandle(ref, () => ({
    getProjectedPoints: () =>
      model.joints.map(joint => {
        const vec = new THREE.Vector3(...joint.position)
        vec.project(camera)
        return { x: vec.x, y: vec.y, z: vec.z }
      }),
  }))

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={1.5} />
      <directionalLight position={[1, 2, 2]} intensity={3} color="#4af0c4" />
      <directionalLight position={[-2, 1, -1]} intensity={1.5} color="#7eb8f7" />
      <pointLight position={[0, -1, -2]} intensity={2} color="#1a4fff" distance={8} />

      {/* Joints */}
      {model.joints.map(joint => (
        <JointMesh
          key={joint.id}
          joint={joint}
          isSelected={selectedJoint?.id === joint.id}
          onPointerClick={selectJoint}
          onPointerDown={handleJointPointerDown}
        />
      ))}

      {/* Bones */}
      {model.bones.map((bone, i) => (
        <BoneLine key={i} bone={bone} />
      ))}

      <OrbitControls ref={orbitRef} enableDamping dampingFactor={0.08} />
      <gridHelper args={[6, 20, '#0d2040', '#0a1828']} position={[0, -1.8, 0]} />
    </>
  )
})

export default Scene
