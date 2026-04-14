// api code here 
import axios, { AxiosError } from 'axios';
import type { PoseType } from '@/types/pose';

const API_URL = `${import.meta.env.VITE_API_URL}`

interface ApiResponse<T> {
    success: boolean; 
    data?: T;
    error?: string;
}

/* ----------------------------- */
// Datasets API 
const datasetClient = axios.create({
    baseURL: `${API_URL}/datasets`,
    headers: {
        'Content-Type': 'application/json'
    },
    timeout:10000,
})



export const datasetApi = {
    // 
}

/* -------------------------- */
// Pose API
const poseClient = axios.create({
    baseURL: `${API_URL}/poses`,
    timeout:10000,
})

export const poseApi = {
    //upload PNG to create Pose Object, returns resulting pose
    async createPoseFromPNG(file: File): Promise<ApiResponse <{pose: PoseType}>>{
        try {
            const formData = new FormData();
            formData.append("file", file);
            
            const {data} = await poseClient.post("/create-from-png", formData)
            return {success: true, data}
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown Error'
            };
        }
    }
}