import * as THREE from 'three'
import { useEffect, useRef } from 'react'
import type { Bone } from "../skeleton/Bone";

interface BoneLineProps {
  bone: Bone
}

export function BoneLine({ bone }: BoneLineProps) {
    const [sx, sy, sz] = bone.start.position
    const [ex, ey, ez] = bone.end.position
    const array = new Float32Array([sx, sy, sz, ex, ey, ez])
    const attrRef = useRef<THREE.BufferAttribute | null>(null)

    useEffect(() => {
      if (attrRef.current) {
        // replace array reference and mark attribute for update
        ;(attrRef.current as any).array = array
        attrRef.current.needsUpdate = true
      }
    }, [sx, sy, sz, ex, ey, ez, array])

    return (
        <line>
        <bufferGeometry>
            <bufferAttribute
            ref={attrRef}
            attach="attributes-position"
            array={array}
            count={2}
            itemSize={3}
            />
        </bufferGeometry>
        <lineBasicMaterial color="#7eb8f7" transparent opacity={0.65} />
        </line>
    )
}