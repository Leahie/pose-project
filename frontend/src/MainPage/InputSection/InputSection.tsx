import FileUpload from "@/MainPage/components/fileUpload";

export default function InputSection(){
    return(
        <div className="full-box">
            {/* Input section can contain Upload component and 3D modeler component */}

            <FileUpload />
        </div>
    )
}