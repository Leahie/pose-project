// api code here 
import axios, { AxiosError } from 'axios';

const API_URL = `${import.meta.env.VITE_API_URL}`

/* ----------------------------- */
// Datasets API 
const datasetClient = axios.create({
    baseURL: `${API_URL}/datasets`,
    headers: {
        'Content-Type': 'application/json'
    },
    timeout:10000,
})

interface ApiResponse<T> {
    success: boolean; 
    data?: T;
    error?: string;
}

export const datasetApi = {

}

/* -------------------------- */
// 