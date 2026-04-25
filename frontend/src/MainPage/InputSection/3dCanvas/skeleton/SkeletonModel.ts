import type { Joint } from './Joint'
import type { Bone } from './Bone'

export class SkeletonModel {
  joints: Joint[]
  bones: Bone[]

  constructor(embedding: [number, number, number][], edges: [number, number][]) {
    this.joints = embedding.map((p, i) => ({
      id: i,
      position: p,
      parent: null,
      children: [],
    }))

    this.bones = []

    edges.forEach(([a, b]) => {
      if (a >= this.joints.length || b >= this.joints.length) return

      const start = this.joints[a]
      const end = this.joints[b]

      if (!end.parent) {
        end.parent = start
        start.children.push(end)
      }

      const dx = end.position[0] - start.position[0]
      const dy = end.position[1] - start.position[1]
      const dz = end.position[2] - start.position[2]
      const length = Math.sqrt(dx * dx + dy * dy + dz * dz)

      this.bones.push({ start, end, length })
    })
  }
}