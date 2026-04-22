import { useCallback, useRef, useState } from "react";
import type { PoseType } from "@/types/pose";

let embedding = 
[
    [
        0.5187111496925354,
        0.3041764795780182,
        -1.088034749031067
    ],
    [
        0.5375338196754456,
        0.26734524965286255,
        -1.0736162662506104
    ],
    [
        0.5485199093818665,
        0.26385730504989624,
        -1.0737046003341675
    ],
    [
        0.556736946105957,
        0.2608981132507324,
        -1.0741997957229614
    ],
    [
        0.504964292049408,
        0.26978278160095215,
        -1.0644859075546265
    ],
    [
        0.4971505403518677,
        0.26793813705444336,
        -1.0645960569381714
    ],
    [
        0.49007129669189453,
        0.2654382586479187,
        -1.0647622346878052
    ],
    [
        0.5761634111404419,
        0.256431519985199,
        -0.8538516163825989
    ],
    [
        0.4856789708137512,
        0.2582128643989563,
        -0.808861255645752
    ],
    [
        0.5391726493835449,
        0.32975518703460693,
        -0.9902424812316895
    ],
    [
        0.505864679813385,
        0.3316868543624878,
        -0.9777287840843201
    ],
    [
        0.6658281087875366,
        0.40515047311782837,
        -0.6629785895347595
    ],
    [
        0.4249085783958435,
        0.35962429642677307,
        -0.530180037021637
    ],
    [
        0.7002454996109009,
        0.6688889861106873,
        -0.644582211971283
    ],
    [
        0.3580923080444336,
        0.5582031607627869,
        -0.3987025022506714
    ],
    [
        0.5691962242126465,
        0.7838932871818542,
        -0.8127426505088806
    ],
    [
        0.32295548915863037,
        0.7241042852401733,
        -0.5467113852500916
    ],
    [
        0.5344533920288086,
        0.8365802764892578,
        -0.9038532376289368
    ],
    [
        0.2940119504928589,
        0.7730042934417725,
        -0.5987404584884644
    ],
    [
        0.5128669738769531,
        0.806913435459137,
        -0.9075586199760437
    ],
    [
        0.31342750787734985,
        0.780240535736084,
        -0.6624671816825867
    ],
    [
        0.5189058184623718,
        0.7897339463233948,
        -0.8146132230758667
    ],
    [
        0.32820630073547363,
        0.7668774127960205,
        -0.5749174952507019
    ],
    [
        0.5802143812179565,
        0.7718719244003296,
        -0.05301321670413017
    ],
    [
        0.4532466530799866,
        0.7390472888946533,
        0.05351629853248596
    ],
    [
        0.5597785115242004,
        0.9918391704559326,
        -0.018690133467316628
    ],
    [
        0.43541279435157776,
        0.9718493223190308,
        0.21838293969631195
    ],
    [
        0.5610696077346802,
        1.154466152191162,
        0.7480469942092896
    ],
    [
        0.44461604952812195,
        1.1417896747589111,
        0.9057620763778687
    ],
    [
        0.5647563934326172,
        1.1680444478988647,
        0.8224232792854309
    ],
    [
        0.4562990665435791,
        1.165862798690796,
        0.9751561880111694
    ],
    [
        0.5603018403053284,
        1.254590392112732,
        0.6862180233001709
    ],
    [
        0.4158269762992859,
        1.2475093603134155,
        0.8138422966003418
    ]
]

export default function SkeletonPipeline() {
  // temporary input for experimentation
  const inputData = {
    id: "33ed57a7-05e8-4c2f-bf72-30f972d047ab",
    visibility: [99,99,99,99,99,99,99,99,99,99,99,99,99,99,97,98,94,95,89,95,90,95,90,99,99,13,8,0,0,3,1,0,1],
    metadata: { created_at: "2026-04-22", title: "000041029.jpg" }
  };
 
//   // Build visibility map: index → value
//   const [visibility, setVisibility] = useState(() => {
//     const map = {};
//     inputData.visibility.forEach((v, i) => { map[i] = v; });
//     return map;
//   });
 
  const [joints, setJoints] = useState(embedding);
 
  const [camera, setCamera] = useState({
    rotX: 0.1, rotY: 0.2, panX: 0, panY: 0, zoom: 280,
  });
 
  const [mode, setMode] = useState("rotate"); // rotate | pan | edit
  const [editAxis, setEditAxis] = useState("xy"); // xy | x | y | z
  const [highlighted, setHighlighted] = useState(null);
 
  // Interaction state
  const drag = useRef(null);
  const canvasWrapRef = useRef(null);
 
 
  // ── Mouse on canvas ──
  const getJointAtScreen = useCallback((mx, my) => {

  }, [joints, camera]);
 
  const onMouseDown = useCallback((e) => {

  }, [mode, camera, joints, getJointAtScreen]);
 
  const onMouseMove = useCallback((e) => {

    }
  }, [camera, editAxis]);
 
  const onMouseUp = useCallback(() => { drag.current = null; }, []);
 
  const onWheel = useCallback((e) => {
  }, []);
 

  return (
    <div >
      {/* Header */}
      <div >

      </div>
 
      {/* Main layout */}
      <div style={{ display: "flex", flex: 1, gap: 0 }}>
        {/* Canvas area */}
        <div
          ref={canvasWrapRef}
          style={{ flex: 1, position: "relative", cursor: mode === "edit" ? "crosshair" : mode === "pan" ? "grab" : "default" }}
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
          onWheel={onWheel}
        >
          <SkeletonCanvas
            joints={joints}
            visibility={visibility}
            camera={camera}
            mode={mode}
            editAxis={editAxis}
            highlighted={highlighted}
          />
 
        </div>
 
        
      </div>
 
     
    </div>
  );
}