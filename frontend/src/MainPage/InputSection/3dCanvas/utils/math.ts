import * as THREE from 'three'

/* Ensure v is within min and max */
export function clamp(v: number, min: number, max:number):number{
    return Math.min(max, Math.max(v, min))
}

/* Linear interpolation for smooth rotation */
export function lerp(a: number, b: number, t: number){
   return a + (b-a) * t
}

/** Convert degrees to radians. */
export function deg2rad(degrees: number): number {
  return degrees * (Math.PI / 180)
}

/** Convert radians to degrees. */
export function rad2deg(radians: number): number {
  return radians * (180 / Math.PI)
}
 
/** 
 * Decompose a quaternion into swing and twist 
 * Mathematically subtracting the twist from the combined Quaternion
 */
export function swingTwistDecompose(
    q: THREE.Quaternion, 
    twistAxis: THREE.Vector3
):[THREE.Quaternion, THREE.Quaternion]{
    const r = new THREE.Vector3(q.x, q.y, q.z)
    const proj = twistAxis.clone().multiplyScalar(r.dot(twistAxis))
    const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, q.w).normalize()
    const swing = q.clone().multiply(twist.clone().invert())
    return [swing, twist]
}

/** Compute the angle (radians) between two 3-vectors. */
export function angleBetween(a: THREE.Vector3, b: THREE.Vector3): number {
  return Math.acos(clamp(a.clone().normalize().dot(b.clone().normalize()), -1, 1))
}

/** Get the cords through the camera into the screen cords  */
export function projectToNDC(
  worldPos: THREE.Vector3,
  camera: THREE.Camera,
): THREE.Vector3 {
  return worldPos.clone().project(camera)
}
 
/** Get the squared distance between coordinates */
export function distSq(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2
}