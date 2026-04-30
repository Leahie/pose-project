import * as THREE from 'three'

export type JointType = 
| 'free'
| 'hinge'
| 'ball'
| 'fixed'
export interface Joint {
  id: number
  position: [number, number, number]
  parent: Joint | null
  children: Joint[]
  localRotation: THREE.Euler
  restOffset: THREE.Vector3
  boneLength: number
  jointType: JointType
}