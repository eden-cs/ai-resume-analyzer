// Shape of a single importance level in the feedback object
export interface FeedbackLevel {
    message: string         // feedback message for this importance level   
    keywords: string[]      // list of missing keywords at this importance level
}

// Shape of the full feedback object returned by the API
export interface Feedback {
    high: FeedbackLevel 
    medium: FeedbackLevel
    low: FeedbackLevel
}

// Shape of missing keyowrds categorized by importance level
export interface MissingKeywords {
    high: string[]
    medium: string[]
    low: string[]
}

// Shape of the full analysis result returned by the API
export interface AnalysisResult {
    match_score: number
    missing_keywords: MissingKeywords
    summary: string
    feedback: Feedback
    suggestions: string
}

// Shape of the form validation errors on upload page
export interface AnalysisError {
    resumeError: boolean            // true if no resume was uploaded
    jobDescriptionError: boolean    // true if no job description was entered
}