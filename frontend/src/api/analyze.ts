import axios from "axios"
import type { AnalysisResult } from "../types"

// Send resume file and job description to the FastAPI backend for analysis
export async function analyzeResume(
    resumeFile: File,
    jobDescription: string
): Promise<AnalysisResult> {

    // Build form data
    const formData = new FormData()
    formData.append("file", resumeFile)
    formData.append("job_desc", jobDescription)

    try {
        // POST request to analyze endpoint, expecting an AnalysisResult in response
        const response = await axios.post<AnalysisResult>(
            `${import.meta.env.VITE_API_URL}/analyze`,
            formData
        )
        return response.data

    } catch (error) {
        console.error("API call failed:", error)
        throw error
    }
}