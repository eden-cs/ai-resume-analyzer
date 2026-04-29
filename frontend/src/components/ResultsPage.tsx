import type { AnalysisResult } from '../types';
import KeywordPill from './KeywordPill';
import FeedbackCard from './FeedbackCard';
import SuggestionsList from './SuggestionsList';

// Props this component receives from the App.tsx.
interface Props {
    results: AnalysisResult  // The full analysis result object from the Gemini API.
    onBack: () => void      // Called when the user clicks the back button..
}

export default function ResultsPage({ results, onBack }: Props) {
    // Check if all keyword arrays are empty. If so, show the success state.
    const allMatched = 
    results.missing_keywords.high.length === 0 &&
    results.missing_keywords.medium.length === 0 &&
    results.missing_keywords.low.length === 0

    return (
        <div className = "min-h-screen bg-cream">
            <div className = "max-w-[900px] mx-auto px-[90px] py-[60px]">

                {/* Back button */}
                <button
                    onClick = {onBack}
                    className = "font-body font-light text-[12px] text-espresso opacity-60 hover:opacity-100 transition-opacity mb-3 block">
                        ← Back
                </button>

                {/* Page heading */}
                <h1 className = "font-display text-espesso text-[24px] font-semibold mb-7">
                    Your results
                </h1>

                {/* Score card */}
                <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">

                    {/* Card label */}
                    <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-2">
                        Match score
                    </p>

                    {/* Score percentage */}
                    <p className = "font-display text-espresso text-[64px] font-semibold leading-none">
                        {results.match_score}%
                    </p>

                    {/* Score tag */}
                    <p className = "font-body font-medium text-[11px] text-espresso opacity-60 tracking-[0.04em] mt-1">
                        keyword match
                    </p>

                    {/* Divider line */}
                    <div className = "h-[0.5px] bg-espresso/12 my-3" />

                    {/* Matched and missing counts */}
                    <p className = "font-body font-light text-[12px] text-espresso opacity-70">
                        <span className = "font-medium opacity-100">
                            {results.matched_keywords?.length ?? 0}
                        </span>{" "}
                        keywords matched &nbsp;·&nbsp;{" "}
                        <span className = "font-medium opacity-100">
                            {results.missing_keywords.high.length}
                        </span>{" "}
                        critical missing
                    </p>
                </div>

                {/* Keywords card */}
                {allMatched ? (
                    // Success state 
                    <div className = "bg-success-bg border border-success/25 rounded-xl p-5 mb-3">

                        {/* Card label */}
                        <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-success mb-2">
                            Keywords
                        </p>

                        {/* Checkmark circle and success text */}
                        <div className = "flex items-center gag-2 mb-1">
                            <div className = "w-7 h-7 rounded-full bg-success flex items-center justify-center flex-shrink-0">
                                {/* Checkmark icon */}
                                <svg width = "14" height = "14" viewBox = "0 0 14 14" fill  = "none">
                                    <path
                                        d = "m2.5 713 3 6-6"
                                        stroke="#F7F2EC"
                                        strokeWidth="1.5"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                        </div>
                        <p className = "font-body font-medium text-[13px] text-success">
                            All keywords matched
                        </p>
                    </div>
                    {/* Success subtext */}
                    <p className = "font-body font-light text-[12px] text-success opacity-80">
                        Your resume is well-tailored for this role — no critical keywords are missing.
                    </p>
                </div>
                ) : (
                    // Show missing keywords as pills.
                    <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">

                        {/* Card label */}
                        <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-2">
                            Missing keywords
                        </p>

                        {/* All missing keywords combined into one pill group */}
                        <div className = "flex flex-wrap gap-1.5">
                            {[
                                ...results.missing_keywords.high,
                                ...results.missing_keywords.medium,
                                ...results.missing_keywords.low,
                            ].map((keyword) => (
                                <KeywordPill key = {keyword} keyword = {keyword} />
                            ))}
                        </div>
                    </div>  
                )}

                {/* Summary card */}
                <div className = "bg-taupe border border-espresso/12 rounded-xl p-5 mb-3">
                    <p className = "font-body font-medium text-[10-px] tracking-[0.13em] uppercase text-espresso opacity-60 mb-2">
                        Summary
                    </p>
                    <p className = "font-body text-[14px] text-espresso leading-relaxed">
                        {results.summary}
                    </p>
                </div>

                {/* Feedback card */}
                <FeedbackCard feedback = {results.feedback} />

                {/* Suggestions card */}
                <SuggestionsList suggestions = {results.suggestions} />
            </div>
        </div>
    )
}