import { useState } from "react";
import type { AnalysisResult } from "./types";
import UploadPage from "./components/UploadPage";
import ResultsPage from "./components/ResultsPage";

export default function App() {
  // Track which page is currently showing
  const [page, setPage] = useState<"upload" | "results">("upload");

  // Store the analysis results from the API (null until the analysis is complete)
  const [results, setResults] = useState<AnalysisResult | null>(null);

  // Tracks whether the API call is in progress
  const [loading, setLoading] = useState(false);

  // Called when analysis succeeds: saves results and switched to results page
  const handleResults = (data: AnalysisResult) => {
    setResults(data);
    setPage("results");
  };

  // Called when user clicks back: clears results and returns to upload page
  const handleBack = () => {
    setPage("upload");
    setResults(null);
  };

  // Show loading screen while API call is in progress
  if (loading) {
    return (
      <div className="min-h-screen bg-cream flex items-center justify-ceneter">
        <div className="text-center">
          <p className="font-display text-espresso text-[24px] font-semibold mb-2">
            Analyzing your resume...
          </p>
          <p className="font-body font-light text-[14px] text-espresso opactiy-60">
            This usually takes a few seconds
          </p>
        </div>
      </div>
    );
  }

  // Show upload page or results page based on current state
  return (
    <div>
      {page == "upload" ? (
        <UploadPage
          onResults={handleResults}
          loading={loading}
          setLoading={setLoading}
        />
      ) : (
        <ResultsPage results={results!} onBack={handleBack} />
      )}
    </div>
  );
}
