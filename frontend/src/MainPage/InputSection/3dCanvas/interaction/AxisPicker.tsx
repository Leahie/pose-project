import type { GizmoAxis } from "./useGizmo";


interface AxisPickerProps {
  activeAxis: GizmoAxis
  setActiveAxis: (axis: GizmoAxis) => void
}
 
export function AxisPicker({ activeAxis, setActiveAxis }: AxisPickerProps) {
  const axes: { axis: GizmoAxis; color: string; label: string }[] = [
    { axis: 'x', color: '#ff4455', label: 'X' },
    { axis: 'y', color: '#44ff88', label: 'Y' },
    { axis: 'z', color: '#4488ff', label: 'Z' },
  ]
 
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      {axes.map(({ axis, color, label }) => (
        <button
          key={axis}
          onClick={() => setActiveAxis(activeAxis === axis ? null : axis)}
          style={{
            width: 36, height: 36,
            borderRadius: '50%',
            border: `2px solid ${color}`,
            background: activeAxis === axis ? color : 'rgba(0,0,0,0.6)',
            color: activeAxis === axis ? '#000' : color,
            fontWeight: 700,
            fontSize: 14,
            cursor: 'pointer',
            transition: 'all 0.15s',
          }}
        >
          {label}
        </button>
      ))}
    </div>
  )
}