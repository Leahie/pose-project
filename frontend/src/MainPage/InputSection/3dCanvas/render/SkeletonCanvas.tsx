import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { forwardRef, useImperativeHandle, useMemo } from 'react'
import { SkeletonModel } from '../skeleton/SkeletonModel'
import { JointMesh } from './JointMesh'
import { BoneLine } from './BoneLine'
import type { SkeletonCanvasRef } from '../../types'
import * as THREE from 'three'

const embedding: [number, number, number][] = [
  [0.5187111496925354, 0.3041764795780182, -1.088034749031067],
  [0.5375338196754456, 0.26734524965286255, -1.0736162662506104],
  [0.5485199093818665, 0.26385730504989624, -1.0737046003341675],
  [0.556736946105957, 0.2608981132507324, -1.0741997957229614],
  [0.504964292049408, 0.26978278160095215, -1.0644859075546265],
  [0.4971505403518677, 0.26793813705444336, -1.0645960569381714],
  [0.49007129669189453, 0.2654382586479187, -1.0647622346878052],
  [0.5761634111404419, 0.256431519985199, -0.8538516163825989],
  [0.4856789708137512, 0.2582128643989563, -0.808861255645752],
  [0.5391726493835449, 0.32975518703460693, -0.9902424812316895],
  [0.505864679813385, 0.3316868543624878, -0.9777287840843201],
  [0.6658281087875366, 0.40515047311782837, -0.6629785895347595],
  [0.4249085783958435, 0.35962429642677307, -0.530180037021637],
  [0.7002454996109009, 0.6688889861106873, -0.644582211971283],
  [0.3580923080444336, 0.5582031607627869, -0.3987025022506714],
  [0.5691962242126465, 0.7838932871818542, -0.8127426505088806],
  [0.32295548915863037, 0.7241042852401733, -0.5467113852500916],
  [0.5344533920288086, 0.8365802764892578, -0.9038532376289368],
  [0.2940119504928589, 0.7730042934417725, -0.5987404584884644],
  [0.5128669738769531, 0.806913435459137, -0.9075586199760437],
  [0.31342750787734985, 0.780240535736084, -0.6624671816825867],
  [0.5189058184623718, 0.7897339463233948, -0.8146132230758667],
  [0.32820630073547363, 0.7668774127960205, -0.5749174952507019],
  [0.5802143812179565, 0.7718719244003296, -0.05301321670413017],
  [0.4532466530799866, 0.7390472888946533, 0.05351629853248596],
  [0.5597785115242004, 0.9918391704559326, -0.018690133467316628],
  [0.43541279435157776, 0.9718493223190308, 0.21838293969631195],
  [0.5610696077346802, 1.154466152191162, 0.7480469942092896],
  [0.44461604952812195, 1.1417896747589111, 0.9057620763778687],
  [0.5647563934326172, 1.1680444478988647, 0.8224232792854309],
  [0.4562990665435791, 1.165862798690796, 0.9751561880111694],
  [0.5603018403053284, 1.254590392112732, 0.6862180233001709],
  [0.4158269762992859, 1.2475093603134155, 0.8138422966003418],
]

const edges: [number, number][] = [
  [8, 7], [10, 9], [8, 10], [7, 9],
  [12, 11], [24, 23], [12, 24], [11, 23],
  [24, 26], [26, 28],
  [23, 25], [25, 27],
  [12, 14], [14, 16],
  [11, 13], [13, 15],
  [16, 22], [16, 20], [20, 18], [18, 16],
  [15, 21], [15, 19], [19, 17], [17, 15],
  [28, 32], [32, 30], [30, 28],
  [27, 31], [31, 29], [29, 27],
]

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