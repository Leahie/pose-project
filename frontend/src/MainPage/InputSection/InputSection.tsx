import FileUpload from "@/MainPage/components/FileUpload";
import { SkeletonCanvas } from "@/MainPage/InputSection/3dCanvas/render/SkeletonCanvas";

export default function InputSection(){
    return(
        <div className="full-box">
            {/* Input section can contain Upload component and 3D modeler component */}
            <SkeletonCanvas />
            <FileUpload />
        </div>
    )
}