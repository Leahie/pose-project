import FileUpload from "@/MainPage/components/FileUpload";
import { SkeletonCanvas } from "@/MainPage/InputSection/3dCanvas/render/SkeletonCanvas";
import { useRef, useState } from "react";
import type { SkeletonCanvasRef } from './types'


export default function InputSection(){
    const canvasRef = useRef<SkeletonCanvasRef>(null)
    const [projectedData, setProjectedData] = useState<{x: number, y: number, z: number}[] | null>(null)

    const handleCalculateView =  () => {
        if (canvasRef.current){
            setProjectedData(canvasRef.current.getProjectedPoints());
        }
    }
    return(
        <div className="full-box relative">
            {/* Input section can contain Upload component and 3D modeler component */}
            <SkeletonCanvas ref={canvasRef}/>

            {/* UI Overlay for calculating projection */}
      <div style={{ position: "absolute", bottom: "20px", left: "20px", color: "white", zIndex: 10 }}>
        <button 
          onClick={handleCalculateView}
          style={{ padding: "10px 20px", cursor: "pointer", background: "#1a4fff", color: "white", border: "none", borderRadius: "4px" }}
        >
          Recalculate View Embedding
        </button>
        
        {projectedData && (
          <div style={{ 
            marginTop: "10px", 
            background: "rgba(0,0,0,0.7)", 
            padding: "10px", 
            maxHeight: "200px", 
            overflowY: "auto",
            borderRadius: "4px",
            fontFamily: "monospace",
            fontSize: "12px"
          }}>
            {projectedData.map((pt, i) => (
              <div key={i}>
                Joint {i}: [{pt.x.toFixed(3)}, {pt.y.toFixed(3)}, {pt.z.toFixed(3)}]
              </div>
            ))}
          </div>
        )}
      </div>

            <FileUpload />
        </div>
    )
}