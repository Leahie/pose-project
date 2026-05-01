import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { forwardRef, useImperativeHandle, useMemo } from 'react'
import { SkeletonModel } from '../skeleton/SkeletonModel'
import { JointMesh } from './JointMesh'
import { BoneLine } from './BoneLine'
import type { SkeletonCanvasRef } from '../../types'
import * as THREE from 'three'
import { edges, embedding } from '../data'

interface SceneProps {
  model: SkeletonModel;
}

const Scene = forwardRef<SkeletonCanvasRef, SceneProps>(({model}, ref) => {
  const { camera } = useThree()  
  const cx = model.joints.reduce((s, j) => s + j.position[0], 0) / model.joints.length
    const cy = model.joints.reduce((s, j) => s + j.position[1], 0) / model.joints.length
    const cz = model.joints.reduce((s, j) => s + j.position[2], 0) / model.joints.length

    const centeredModel = useMemo(() => {
        const scale = 3
        const centeredEmbedding: [number, number, number][] = model.joints.map(j => [
        (j.position[0] - cx) * scale,
        (j.position[1] - cy) * scale,
        (j.position[2] - cz) * scale,
        ])
        return new SkeletonModel(centeredEmbedding, edges)
    }, [model, cx, cy, cz])

    useImperativeHandle(ref, () => ({
      getProjectedPoints: () => {
          return centeredModel.joints.map(joint => {
            const vec = new THREE.Vector3(...joint.position);
            vec.project(camera);
            return { x: vec.x, y: vec.y, z: vec.z };
          })
      } 
    }) 

    )
    return (
    <>
      {/* Lighting */}
      <ambientLight intensity={1.5} />
      <directionalLight position={[1, 2, 2]} intensity={3} color="#4af0c4" />
      <directionalLight position={[-2, 1, -1]} intensity={1.5} color="#7eb8f7" />
      <pointLight position={[0, -1, -2]} intensity={2} color="#1a4fff" distance={8} />

      {/* Joints */}
      {centeredModel.joints.map(joint => (
        <JointMesh key={joint.id} joint={joint} />
      ))}

      {/* Bones */}
      {centeredModel.bones.map((bone, i) => (
        <BoneLine key={i} bone={bone} />
      ))}
    </>
  )
})

export const SkeletonCanvas = forwardRef<SkeletonCanvasRef>((props, ref) => {
  const model = useMemo(() => new SkeletonModel(embedding, edges), [])

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050810' }}>
      <Canvas camera={{ position: [0, 0, 3.5], fov: 55 }}>
        <Scene model={model} ref={ref}/>
        <OrbitControls enableDamping dampingFactor={0.08} />
        <gridHelper args={[6, 20, '#0d2040', '#0a1828']} position={[0, -1.8, 0]} />
      </Canvas>
    </div>
  )
})