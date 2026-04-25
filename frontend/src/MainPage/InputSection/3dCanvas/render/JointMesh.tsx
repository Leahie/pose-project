import { useRef } from 'react'
import { Mesh } from 'three'
import type { Joint } from '../skeleton/Joint'

interface JointMeshProps {
  joint: Joint
}

export function JointMesh({ joint }: JointMeshProps) {
    const ref = useRef<Mesh>(null)
    const [x,y,z] = joint.position
    return(
        <mesh ref={ref} position={[x,y,z]}>
            <sphereGeometry args={[0.045, 16, 16]} />
            <meshStandardMaterial
                color="#4af0c4"
                emissive="#0a4030"
                roughness={0.2}
                metalness={0.8}
            />
        </mesh>
    )
}