import { useState } from "react";
import type { AnalysisResult, AnalysisError } from "../types";
import DropZone from "./DropZone";
import ErrorMessage from "./ErrorMessage";
import { analyzeResume } from "../api/analyze";

// Props received from App.tsx
interface Props {
  onResults: (data: AnalysisResult) => void; // Called when analysis succeeds
  loading: boolean; // Indicates if analysis is in progress
  setLoading: (loading: boolean) => void; // Function to update loading state
}

export default function UploadPage({ onResults, loading, setLoading }: Props) {
  // Stores the uploaded resume rule (null until the user uploads their resume)
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  // Stores the job description text as user types
  const [jobDescription, setJobDescription] = useState("");

  // Tracks which fields have validation errors
  const [errors, setErrors] = useState<AnalysisError>({
    resumeError: false,
    jobDescriptionError: false,
  });

  // Validates inputs and calls the API when the user clicks Analyze now
  const handleAnalyze = async () => {
    // Build error object by checking each field
    const newErrors = {
      resumeError: !resumeFile,
      jobDescriptionError: !jobDescription.trim(),
    };

    // Update errors state so UI shows error warnings
    setErrors(newErrors);

    // Stop here if any field is invalid (don't call the API)
    if (newErrors.resumeError || newErrors.jobDescriptionError) {
      return;
    }

    // If both fields are valid, call the API
    try {
      setLoading(true);
      const data = await analyzeResume(resumeFile!, jobDescription);

      // Pass results up to App.tsx, which will swtich to the results page
      onResults(data);
    } catch (error) {
      console.error("Analysis failed:", error);
    } finally {
      // Always turn off loading whether API call succceeded or failed
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-cream">
      {/* Centered container - max 900px wide */}
      <div className="max-w-[900px] mx-auto px-[90px] py-[60px]">
        {/* Eyebrow label */}
        <p className="font-body font-medium text-[11px] tracking-[0.13em] uppercase text-terracotta mb-2">
          AI Resume Analyzer
        </p>

        {/* Heading */}
        <h1 className="font-display text-espresso text-[32px] font-semibold leading-[1.22] mb-4">
          Your resume, <br />
          <em> refined.</em>
        </h1>

        {/* Subtitle */}
        <p className="font-body font-light text-[14px] text-espresso opacity-75 leading-[1.65] mb-10">
          Upload your resume, paste the job description, and <br />
          get instant AI-powered feedback tailored to your role!
        </p>

        {/* Resume field label */}
        <p className="font-body font-medium text-[10px] tracking-[0.11em] uppercase text-espresso opacity-60 mb-[10px]">
          Your Resume
        </p>

        {/* DropZone component */}
        <DropZone
          file={resumeFile}
          onFileChange={setResumeFile}
          hasError={errors.resumeError}
        />

        {/* Only renders if resumeError is true */}
        {errors.resumeError && (
          <ErrorMessage message="Resume is required to analyze" />
        )}

        {/* Job description field label */}
        <p className="font-body font-medium text-[10px] tracking-[0.11em] uppercase text-espresso opacity-60 mb-[10px] mt-5">
          Job Description
        </p>

        {/* Job description textarea */}
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Paste the job description here..."
          rows={7}
          className={`w-full rounded-xl p-4 font-body font-light text-[14px] text-espresso resize-none outline-none leading-relaxed placeholder:text-espresso placeholder:opacity-40 ${
            errors.jobDescriptionError
              ? "bg-error-bg border border-error"
              : "bg-taupe border border-espresso/10"
          }`}
        />

        {/* Only renders if jobDescriptionError is true */}
        {errors.jobDescriptionError && (
          <ErrorMessage message="Job description is required to analyze" />
        )}

        {/* Analyze now button */}
        <div className="flex justify-center mt-6">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="bg-dark-cream border border-espresso/25 text-espresso font-body font-medium text-[15px] px-12 py-3 rounded-xl hover:opacity-80 transition-opacity disabled:opacity-50"
          >
            {loading ? "Analyzing..." : "Analyze now"}
          </button>
        </div>
      </div>
    </div>
  );
}
