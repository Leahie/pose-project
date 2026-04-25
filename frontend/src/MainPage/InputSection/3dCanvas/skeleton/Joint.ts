export interface Joint {
  id: number
  position: [number, number, number]
  parent: Joint | null
  children: Joint[]
}