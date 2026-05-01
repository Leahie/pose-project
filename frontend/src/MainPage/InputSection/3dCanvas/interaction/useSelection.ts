import { useState, useCallback } from 'react'
import type { Joint } from '../skeleton/Joint'

export interface UseSelectionReturn {
  selectedJoint: Joint | null
  selectJoint: (joint: Joint | null) => void
  clearSelection: () => void
}

export function useSelection(): UseSelectionReturn {
  const [selectedJoint, setSelectedJoint] = useState<Joint | null>(null)
  const selectJoint = useCallback((joint: Joint | null) => {
    setSelectedJoint(joint)
  }, [])
  const clearSelection = useCallback(() => {
    setSelectedJoint(null)
  }, [])
 
  return { selectedJoint, selectJoint, clearSelection }

}