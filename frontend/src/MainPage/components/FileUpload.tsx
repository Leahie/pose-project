import {useState, } from "react";
import type {ChangeEvent} from "react";
import { poseApi } from "@/api";
export default function FileUpload(){
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
		if (event.target.files) setSelectedFile(event.target.files[0]);
	};

	const onFileUpload = () => {
        if (!selectedFile) return
		console.log(selectedFile);
		poseApi.createPoseFromPNG(selectedFile)
	};
	const fileData = () => {
		if (selectedFile) {
			return (
				<div>
					<h2>File Details:</h2>
					<p>File Name: {selectedFile.name}</p>
					<p>File Type: {selectedFile.type}</p>
					<p>
						Last Modified: {selectedFile.lastModified.toString()}
					</p>
				</div>
			);
		} else {
			return (
				<div>
					<br />
					<h4>Choose before Pressing the Upload button</h4>
				</div>
			);
		}
	};

    return(
        <div className="block">

			<div className="flex gap-2">
				<input className="bg-slate-200" type="file" onChange={onFileChange} />
				<button className="bg-slate-200 cursor-pointer" onClick={onFileUpload}>Upload!</button>
			</div>
			{fileData()}
		</div>
    )
}