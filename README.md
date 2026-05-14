# Ai Resume Analyzer
An AI-powered resume analyzer that compares your resume against a job description and provides prioritized,actionable feedback to help you land the job.

## Live Demo
https://ai-resume-analyzer-six-dun.vercel.app/

## Figma Prototype
https://www.figma.com/proto/JSByNFcKxrvrbd5FEYvfbm/AI-Resume-Analyzer?node-id=1-8&t=QGtUh0799dIVufXc-1

## Features
- Upload PDF resume
- Paste any job description
- NLP-based keyword matching wiht percentage match score
- Missing keyowrds categorize by importance: high, medium, low
- AI-generate suggestions powered by Gemini API
- Success state with all keywords are matched

## Tech Stack

**Frontend**
- React + TypeSCript
- Tailwind CSS
- Vite + Axios

**Backend**
- Python + FastAPI
- spaCy en_core_web_md
- PyMuPDF
- Google Gemini API

**Infrastructure**
- Docker
- Render
- Vercel

## Known Limitations
- Keyword extraction may include generic nouns from common job description phrases, such as "attention to detail" or "team environment".
- Future improvement: WordNet frequency filtering or noun chunk extraction to reduce noise.