import { useMemo } from "react";
import * as THREE from 'three'
import type { Bone } from "../skeleton/Bone";

interface BoneLineProps {
  bone: Bone
}

export function BoneLine({ bone }: BoneLineProps) {
    const points = useMemo(() => {
        const [sx, sy, sz] = bone.start.position
        const [ex, ey, ez] = bone.end.position
        return [new THREE.Vector3(sx, sy, sz), new THREE.Vector3(ex, ey, ez)]
    }, [bone])

    return (
        <line>
        <bufferGeometry>
            <bufferAttribute
            attach="attributes-position"
            array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
            count={2}
            itemSize={3}
            />
        </bufferGeometry>
        <lineBasicMaterial color="#7eb8f7" transparent opacity={0.65} />
        </line>
    )
}