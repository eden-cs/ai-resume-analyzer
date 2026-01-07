from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz # PyMuPDF
import spacy # NLP library
from collections import Counter

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
        
    # Return the set of short tokens from resume and job description
    return short_resume_tokens, short_job_desc_tokens

def keywords_frequency(resume_text: str, job_desc_text: str, resume_rescued: set, job_desc_rescued: set) -> dict:
    """
    This helper function counts how many times each keyword appears in the resume_text and job_desc_text.

    @param resume_text: str, the resume text.

    @param job_desc_text: str, the job description text.

    @param resume_rescued: set, the set of rescued short tokens from the resume.

    @param job_desc_rescued: set, the set of rescued short tokens from the job description.

    @return: dict, a dictionary with keywords as the keys and the frequency counts as the values.
    """

    resume_keywords = list()
    job_desc_keywords = list()

    # Process the resume and job description text with spaCy
    resume_doc = nlp(resume_text)
    job_desc_doc = nlp(job_desc_text)

    # Loop through tokens in resume text to filter out stop words, punctuations, and numbers
    for token in resume_doc:
        if(token.is_stop):
            continue
        if(token.is_punct or not(token.is_alpha)):
            continue

        # Handle short tokens only if they are in the rescued set
        if(token.lemma_.lower() in resume_rescued):
            resume_keywords.append(token.lemma_.lower())
            continue

        # Ignore short tokens not in rescued set
        if(len(token.lemma_.lower()) <= 2):
            continue

        # Filter out named entities
        if(token.ent_type_ in ["PERSON", "GPE", "DATE", "TIME", "MONEY", "PERCENT"]):
            continue
        
        # Filter out non-relevant parts of speech
        if((token.pos_ != "NOUN") and (token.pos_ != "PROPN") and (token.pos_ != "VERB")):
            continue

        resume_keywords.append(token.lemma_.lower())

    # Loop through tokens in job description text to filter out stop words, punctuations, numbers, non-relevant parts of speech, and named entities
    for token in job_desc_doc:
        if(token.is_stop):
            continue
        if(token.is_punct or not(token.is_alpha)):
            continue

        # Handle short tokens only if they are in the rescued set
        if(token.lemma_.lower() in job_desc_rescued):
            job_desc_keywords.append(token.lemma_.lower())
            continue

         # Ignore short tokens not in rescued set
        if(len(token.lemma_.lower()) <= 2):
            continue

        # Filter out named entities
        if(token.ent_type_ in ["PERSON", "GPE", "DATE", "TIME", "MONEY", "PERCENT"]):
            continue

        # Filter out non-relevant parts of speech
        if((token.pos_ != "NOUN") and (token.pos_ != "PROPN") and (token.pos_ != "VERB")):
            continue
    
        job_desc_keywords.append(token.lemma_.lower())

    # Count frequency of each keyword using Counter
    resume_keyword_freq = Counter(resume_keywords)
    job_desc_keyword_freq = Counter(job_desc_keywords)

    return resume_keyword_freq, job_desc_keyword_freq

def matched_keywords(resume_keyword_freq: dict, job_desc_keyword_freq: dict) -> set:
    """
    This helper function finds the matched keywords between the resume and job description.

    @param resume_keyword_freq: dict, the dictionary of keywords extracted from the resume with their frequencies.

    @param job_desc_keyword_freq: dict, the dictionary of keywords extracted from the job description with their frequencies.

    @return: set, a set of matched keywords.
    """

    # Find intersection of resume and job description keywords
    matched = set(resume_keyword_freq.keys()).intersection(set(job_desc_keyword_freq.keys()))
    
    return matched


def missing_keywords(resume_keyword_freq: dict, job_desc_keyword_freq: dict) -> set:
    """
    This helper function finds the missing keywords in the resume compared to the job description.

    @param resume_keyword_freq: dict, the dictionary of keywords extracted from the resume with their frequencies.

    @param job_desc_keyword_freq: dict, the dictionary of keywords extracted from the job description with their frequencies.

    @return: set, a set of missing keywords.
    """

    missing = set(job_desc_keyword_freq.keys()).difference(set(resume_keyword_freq.keys()))
    
    return missing

def match_score(resume_keyword_freq: dict, job_desc_keyword_freq: dict) -> float:
    """
    This helper function calculates the match score (as a percentage) between the resume and job description keywords.

    @param resume_keyword_freq: dict, the dictionary of keywords extracted from the resume with their frequencies.

    @param job_desc_keyword_freq: dict, the dictionary of keywords extracted from the job description with their frequencies.

    @return: float, the match score as a percentage.
    """

    matched_weight = 0
    required_weight = 0

    # Calculate the coverage of resume keywords against job description keywords
    for keyword in job_desc_keyword_freq:
        total_occurrences = job_desc_keyword_freq[keyword]
        # Return the value if they keyword exists, 0 otherwise
        matched_occurrences = resume_keyword_freq.get(keyword, 0)
        covered = min(total_occurrences, matched_occurrences)

        # Calculate weights
        matched_weight += covered
        required_weight += total_occurrences

    # Avoid division by zero
    if(required_weight == 0):
        return 0.0
    
    match_score = round((matched_weight / required_weight) * 100, 2)
    
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

    # Call on helper function to rescue short tokens from resume and job description
    resume_rescued, job_desc_rescued = rescued_short_tokens(resume_text, job_desc)

    # Call on helper function to get resume keywords and job description keywords frequency
    resume_keyword_freq, job_desc_keyword_freq = keywords_frequency(resume_text, job_desc, resume_rescued, job_desc_rescued)
    
    # Call on helper function to find matched keywords
    matched = matched_keywords(resume_keyword_freq, job_desc_keyword_freq)

    # Call on helper function to find missing keywords
    missing = missing_keywords(resume_keyword_freq, job_desc_keyword_freq)

    # Call on helper function to calculate match score
    score = match_score(resume_keyword_freq, job_desc_keyword_freq)

    return {
        "message": "Resume received", 
        "job_desc": job_desc,
        "resume_keywords": sorted(resume_keywords),
        "job_desc_keywords": sorted(job_desc_keywords),
        "resume_rescued": sorted(resume_rescued),
        "job_desc_rescued": sorted(job_desc_rescued),
        "resume_keyword_freq": dict(resume_keyword_freq),
        "job_desc_keyword_freq":  dict(job_desc_keyword_freq),
        "matched_keywords": sorted(matched),
        "missing_keywords": sorted(missing),  
        "match_score": score
    }

@app.get("/")
def root():
    return {
        "status": "API is running"
        }
