from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from analysis.pdf_analysis import extract_pdf_text
from analysis.keyword_analysis import rescued_short_tokens, keywords_frequency, matched_keywords, missing_keywords
from analysis.scoring_analysis import match_score, analysis_summary
from analysis.ai_analysis import generate_suggestions
from utils.feedback import missing_feedback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)   


@app.post("/analyze")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    # Convert UploadFile to raw bytes
    file_bytes = await file.read()

    # Call on helper function to extract the text from the PDF
    resume_text = extract_pdf_text(file_bytes)

    # Call on helper function to rescue short tokens from resume and job description
    resume_rescued, job_desc_rescued = rescued_short_tokens(resume_text, job_desc)

    # Call on helper function to get resume keywords and job description keywords frequency
    resume_keyword_freq, job_desc_keyword_freq = keywords_frequency(resume_text, job_desc, resume_rescued, job_desc_rescued)
    
    # Call on helper function to find matched keywords
    matched = matched_keywords(resume_keyword_freq, job_desc_keyword_freq)

    # Call on helper function to find missing keywords
    missing = missing_keywords(resume_keyword_freq, job_desc_keyword_freq)

    # Convert missing keywords from sets to lists for JSON serialization
    for importance_level in missing:
        missing[importance_level] = list(missing[importance_level])

    # Call on helper function to generate feedback based on missing keywords
    feedback = missing_feedback(missing)

    # Call on helper function to calculate match score
    score = match_score(resume_keyword_freq, job_desc_keyword_freq)

    # Call on helper function to generate analysis summary
    summary = analysis_summary(score, matched, missing)

    # Call on helper function to generate suggestions for improving the resume
    suggestions = generate_suggestions(missing, score)

    return {
        "match_score": score,
        "matched_keywords": sorted(matched),
        "missing_keywords": missing,  
        "summary": summary,
        "feedback": feedback,
        "suggestions": suggestions
    }


@app.get("/")
def root():
    return {
        "status": "API is running"
        }
