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
      <div className="min-h-screen bg-cream flex items-center justify-center">
        <div className="text-center">
          {/* Doc icon */}
          <div className="flex justify-center mb-7">
            <div className="w-12 h-[58px] bg-[#FAF7F2] border border-espresso/12 rounded-[6px] flex flex-col justify-end p-[8px] gap-[4px]"
              style={{ boxShadow: "3px 3px 0 rgba(44,26,14,0.12)" }}>
              <div className="h-[3px] rounded-sm bg-medium-brown" />
              <div className="h-[3px] rounded-sm bg-medium-brown" />
              <div className="h-[3px] w-[60%] rounded-sm bg-medium-brown" />
            </div>
          </div>
          
          {/* Headline */}
          <p className="font-display text-espresso text-[24px] font-semibold mb-2">
            Analyzing your resume...
          </p>

           {/* Subtitle */}
          <p className="font-body font-light text-[14px] text-espresso opacity-60">
            Matching keywords and generating<br />
            personalized feedback for you
          </p>

          {/* Bouncing dots */}
          <div className="flex justify-center gap-2 mt-8">
            <div className="w-[7px] h-[7px] rounded-full bg-medium-brown animate-bounce" style={{ animationDelay: "0ms" }} />
            <div className="w-[7px] h-[7px] rounded-full bg-medium-brown animate-bounce" style={{ animationDelay: "150ms" }} />
            <div className="w-[7px] h-[7px] rounded-full bg-medium-brown animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>

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
