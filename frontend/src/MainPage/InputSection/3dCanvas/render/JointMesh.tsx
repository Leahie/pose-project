import { useRef } from 'react'
import { Mesh } from 'three'
import type { Joint } from '../skeleton/Joint'

interface JointMeshProps {
  joint: Joint
  isSelected?: boolean
  onPointerDown?: (joint: Joint, e: PointerEvent) => void
  onPointerClick?: (joint: Joint) => void
}

export function JointMesh({ joint, isSelected, onPointerDown, onPointerClick  }: JointMeshProps) {
    const ref = useRef<Mesh>(null)
    const [x,y,z] = joint.position

    const color       = isSelected ? '#ffdd44' : '#4af0c4'
    const emissive    = isSelected ? '#664400' : '#0a4030'
    const radius      = isSelected ? 0.065     : 0.045

    return(
        <mesh
            ref={ref}
            position={[x, y, z]}
            onPointerDown={(e) => {
                e.stopPropagation()
                onPointerDown?.(joint, e.nativeEvent)
            }}
            onClick={(e) => {
                e.stopPropagation()
                onPointerClick?.(joint)
            }}
            >
            <sphereGeometry args={[radius, 16, 16]} />
            <meshStandardMaterial
                color={color}
                emissive={emissive}
                roughness={0.2}
                metalness={0.8}
            />
        </mesh>
    )
}