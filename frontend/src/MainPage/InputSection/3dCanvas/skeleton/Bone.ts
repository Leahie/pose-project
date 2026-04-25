import type { Joint } from './Joint'

export interface Bone {
  start: Joint
  end: Joint
  length: number
}