// Type file for shared classes
export interface SkeletonCanvasRef {
  getProjectedPoints: () => { x: number, y: number, z: number }[];
}
