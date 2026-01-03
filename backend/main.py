from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz # PyMuPDF
import spacy # NLP library

# Load the small English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_pdf_text(file_bytes: bytes) -> str:
    
    """
    This helper function extracts text from a PDF file using PyMuPDF.

    @param file_bytes: bytes, the PDF file to extract text from.

    @return: str, the extracted text from the PDF.
    """
    
    # Open the uploaded PDF file
    pdf_doc = fitz.open(stream = file_bytes, filetype = "pdf")
    text = ""

    # Loop through each page and extract text
    for page in pdf_doc:
        # Append the text of each page to the overall text
        text += page.get_text()

    return text

def extract_keywords(text: str) -> set:
    """
    This helper function extracts keywords from the given text using spaCy.

    @param text: str, the text to extract keywords from.

    @return: set, a set of extracted keywords.
    """

    # Process the text with spaCy
    doc = nlp(text)
    keywords = set()
    
    # Loop through tokens and filter out stop words, punctuation, numbers, and non-relevant parts of speech, and named entities
    for token in doc:
        if(token.is_stop):
            continue
        if(token.is_punct or not(token.is_alpha)):
            continue
        if((token.pos_ != "NOUN") and (token.pos_ != "PROPN") and (token.pos_ != "VERB")):
            continue
        if(token.ent_type_ in ["PERSON", "GPE", "DATE", "TIME", "MONEY", "PERCENT"]):
            continue
    
        # Ignore short lemma (less than or equal to 2 characters)
        if(len(token.lemma_.lower()) <= 2):
            continue

        # Ignore lemma that are digits
        if(token.lemma_.lower().isdigit()):
            continue

        # Normalize word to its lemma form and add to keywords set
        keywords.add(token.lemma_.lower())

    return keywords

def rescued_short_tokens(resume_text: str, job_desc_text: str) -> set:
    """
    This helper function resuces the short tokens (less than or equal to 2 characters) identified as keywords.

    @param resume_text: str, the resume text.

    @param job_desc_text: str, the job description text.

    @return: set, a set of rescued short tokens.
    """

    # Process the resume and job description text with spaCy
    resume_doc = nlp(resume_text)
    job_desc_doc = nlp(job_desc_text)

    short_resume_tokens = set()
    short_job_desc_tokens = set()

    # Loop through tokens (less than or equal to 2 characters) and filter out stop words, punctuation, and numbers.
    for token in resume_doc:
        if(len(token.text) <= 2 and not(token.is_stop) and token.is_alpha):
            short_resume_tokens.add(token.lemma_.lower())
        
    # Loop through tokens (less than or equal to 2 characters) and filter out stop words, punctuation, and numbers.
    for token in job_desc_doc:
        if(len(token.text) <= 2 and not(token.is_stop) and token.is_alpha):
            short_job_desc_tokens.add(token.lemma_.lower())
        
    # Return the intersection of short tokens from resume and job description
    return short_resume_tokens.intersection(short_job_desc_tokens)

def matched_keywords(resume_keywords: set, job_desc_keywords: set, rescued: set) -> set:
    """
    This helper function finds the matched keywords between the resume and job description.

    @param resume_keywords: set, the set of keywords extracted from the resume.

    @param job_desc_keywords: set, the set of keywords extracted from the job description.

    @param rescued: set, the set of rescued short tokens.

    @return: set, a set of matched keywords.
    """

    # Find intersection of resume and job description keywords
    matched = resume_keywords.intersection(job_desc_keywords)
    
    return matched.union(rescued)


def missing_keywords(resume_keywords: set, job_desc_keywords: set, rescued: set) -> set:
    """
    This helper function finds the missing keywords in the resume compared to the job description.

    @param resume_keywords: set, the set of keywords extracted from the resume.

    @param job_desc_keywords: set, the set of keywords extracted from the job description.

    @param rescued: set, the set of rescued short tokens.

    @return: set, a set of missing keywords.
    """

    return job_desc_keywords.difference(resume_keywords.union(rescued))

def match_score(matched: set, job_desc_keywords: set) -> float:
    """
    This helper function calculates the match score (as a percentage) between the resume and job description keywords.

    @param matched: set, the set of matched keywords between the resume and job description.

    @param job_desc_keywords: set, the set of keywords extracted from the job description.

    @return: float, the match score as a percentage.
    """

    # Make sure job_desc_keywords is not empty to avoid division by zero
    if(len(job_desc_keywords) == 0):
        return 0.0
    
    match_score = round((len(matched) / len(job_desc_keywords)) * 100, 2)

    return match_score

@app.post("/analyze")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    # Convert UploadFile to raw bytes
    file_bytes = await file.read()

    # Call on helper function to extract the text from the PDF
    resume_text = extract_pdf_text(file_bytes)

    # Call on helper function to extract keywords from the resume text
    resume_keywords = extract_keywords(resume_text)

    # Call on helper function to extract keywords from the job description
    job_desc_keywords = extract_keywords(job_desc)

    # Call on helper function to rescue short tokens
    rescued = rescued_short_tokens(resume_text, job_desc)

    # Call on helper function to find matched keywords
    matched = matched_keywords(resume_keywords, job_desc_keywords, rescued)

    # Call on helper function to find missing keywords
    missing = missing_keywords(resume_keywords, job_desc_keywords, rescued)

    # Call on helper function to calculate match score
    score = match_score(matched, job_desc_keywords)

    return {
        "message": "Resume received", 
        "job_desc": job_desc,
        "resume_keywords": sorted(resume_keywords),
        "job_desc_keywords": sorted(job_desc_keywords),
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),  
        "match_score": score
        }

@app.get("/")
def root():
    return {
        "status": "API is running"
        }
