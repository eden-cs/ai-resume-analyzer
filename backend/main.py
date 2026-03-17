from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz # PyMuPDF
import spacy # NLP library
from collections import Counter
import os
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google Generative AI client with API key from the environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Rate limiting - max requests per day
REQUEST_COUNT = 0
MAX_DAILY_REQUESTS = 20

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


def rescued_short_tokens(resume_text: str, job_desc_text: str) -> tuple:
    """
    This helper function resuces the short tokens (less than or equal to 2 characters) identified as keywords.

    @param resume_text: str, the resume text.

    @param job_desc_text: str, the job description text.

    @return: tuple, a tuple containing sets of rescued short tokens from the resume and job description.
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


def keyword_importance(job_desc_keyword_freq: dict) -> dict:
    """
    This helper function categorizes each keyword in the job description keyword frequency dictionary into high, medium, and low importance based on their frequency.

    @param job_desc_keyword_freq: dict, the dictionary of keywords extracted from the job description with their frequencies.

    @return: dict, a dictionary with the keys being the labels "high", "medium", and "low", and the values being sets of keywords and their respect frequencies that fall into each category.
    """

    # Intialize importance categories
    keyword_importance = {"high": set(), "medium": set(), "low": set()}

    # Sort the job description keywords by frequency in descending order
    sorted_keywords = sorted(job_desc_keyword_freq.items(), key = lambda item: item[1], reverse = True)

    # Determine frequency thresholds: top 25% as high importance, next 50% as medium importance, rest as low importance
    total_keywords = len(sorted_keywords)
    high_cutoff = int(0.25 * total_keywords)
    medium_cutoff = int(0.75 * total_keywords)

    # Categorize keywords based on their frequency
    for idx, (keyword, freq) in enumerate(sorted_keywords):
        # High importance: frequency is in the top 25% of frequencies
        if(idx < high_cutoff):
            keyword_importance["high"].add((keyword, freq))
        # Medium importance: frequency is in the next 50% of frequencies
        elif(idx < medium_cutoff):
            keyword_importance["medium"].add((keyword, freq))
        # Low importance: frequency is in the bottom 25% of frequencies
        else:
            keyword_importance["low"].add((keyword, freq))

    return keyword_importance


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


def missing_keywords(resume_keyword_freq: dict, job_desc_keyword_freq: dict) -> dict:
    """
    This helper function finds the missing keywords in the resume compared to the job description.

    @param resume_keyword_freq: dict, the dictionary of keywords extracted from the resume with their frequencies.

    @param job_desc_keyword_freq: dict, the dictionary of keywords extracted from the job description with their frequencies.

    @return: dict, a dictionary of missing keywords categorized by importance level.
    """
    # Find keywords that are in the job description but not in the resume
    missing = set(job_desc_keyword_freq.keys()).difference(set(resume_keyword_freq.keys()))
                                                       
    # Call on keyword_importance helper to categorize job description keywords by importance
    job_desc_keyword_importance = keyword_importance(job_desc_keyword_freq)

    important_missing = {"high": set(), "medium": set(), "low": set()}

    # Filter missing keywords based on importance levels
    for importance_level, keywords in job_desc_keyword_importance.items():
        for keyword, _freq in keywords:
            if(keyword in missing):
                important_missing[importance_level].add(keyword)

    return important_missing
    

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


def missing_feedback(missing_keywords: dict) -> dict:
    """
    This helper function generates feedback based on the missing keywords categorized by importance level.

    @param: missing_keywords: dict, a dictionary of missing keywords categorized by importance level.

    @return: dict, the feedback messages based on missing keywords.
    """

    feedback = {"high": {"message": "", "keywords": []}, "medium": {"message": "", "keywords": []}, "low": {"message": "", "keywords": []}}

    # Generate feedback messages and keyword sets for each importance level
    for importance_level, keywords in missing_keywords.items():
        # No missing keywords in this importance level
        if(len(keywords) == 0):
            feedback[importance_level]["message"] = (f"No {importance_level} importance keywords are missing.")
            feedback[importance_level]["keywords"] = []

        # There are missing keywords in this importance level 
        else:
            if(importance_level == "high"):
                message = ("These critical skills are missing from your resume and are essential for this role.")
            elif(importance_level == "medium"):
                message = ("These skills are missing, and including them would strengthen your resume.")
            else:
                message = ("These are nice-to-have skills and would improve your resume.")

            feedback[importance_level]["message"] = message
            feedback[importance_level]["keywords"] = missing_keywords[importance_level]    

    return feedback


def analysis_summary(score: float, matched: set, missing: dict) -> str:
    """
    This helper function generates a summary of the analysis based on the match score, matched keywords, and missing keywords.

    @param score: float, the match score as a percentage.

    @param matched: set, a set of matched keywords.

    @param missing: dict, a dictionary of missing keywords categorized by importance level.

    @return: str, the summary of the analysis.
    """

    # Calculate the number of matched keywords and critical missing keywords
    matched_length = len(matched)
    missing_length = len(missing["high"])

    summary = f"Your resume matches {score}% of the keywords in the job description. You have {matched_length} matched keywords and {missing_length} critical missing keywords."
    
    return summary


def generate_suggestions(missing: dict, score: float) -> str:
    """
    This helper function generates suggestions for improving the resume based on the missing keywords and the job description using OpenAI's language model.

    @param missing: dict, a dictionary of missing keywords categorized by importance level.

    @param score: float, the match score as a percentage.

    @return: str, the generated suggestions for improving the resume.
    """

    global REQUEST_COUNT

    # Check if the daily request limit has been reached
    if (REQUEST_COUNT >= MAX_DAILY_REQUESTS):
        return "Sorry, the daily request limit has been reached. Please try again tomorrow."
    
    REQUEST_COUNT += 1
    
    # Format missing keywords for the prompt
    high_missing = ", ".join(missing["high"]) if missing["high"] else "none"
    medium_missing = ", ".join(missing["medium"]) if missing["medium"] else "none"
    
    # Create a prompt for the language model based on the missing keywords and match score
    prompt = f"You are a resume optimization assistant. A resume has been analyzed against a job description, and the match score is {score}%. The following high importance keywords are missing from the resume: {high_missing}. The following medium importance keywords are missing from the resume: {medium_missing}. Provide 3-5 actionable suggestions on how to improve the resume. Focus on adding missing high importance skills. Keep suggestions concise and professional."

    # Try to generate suggestions using the language model with error handling for potential API errors
    try:
        response = model.generate_content(
            prompt, 
            generation_config = {
                "max_output_tokens": 150, # Limit the response to 150 tokens to ensure concise suggestions
                "temperature": 0.7, # Set temperature to 0.7 for a balance between creativity and relevance
                "top_p": 0.9 # Set top_p to 0.9 to consider the top 90% of token probabilities for generating suggestions
            }
        )

        # Generate suggestions using the language model
        suggestions = response.text.strip() if response and response.text else ""
        return suggestions
    
    except google_exceptions.InvalidArgument as e:
        print("Invalid request:", str(e))

    except google_exceptions.ResourceExhausted as e:
        print("Quota exceeded or rate limit hit. Please try again tomorrow:", str(e))

    except google_exceptions.GoogleAPIError as e:
        print("API call failed:", str(e))

    except Exception as e:
        print("An unexpected error occurred:", str(e))

    # Return feedback based on missing keywords if API call fails
    fallback_suggestions = f"Focus on adding these missing high importance skills to your resume: {high_missing}. Also consider including these medium importance skills: {medium_missing}." 
    return fallback_suggestions
    


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
        "feedback": feedback,
        "summary": summary,
        "suggestions": suggestions
    }


@app.get("/")
def root():
    return {
        "status": "API is running"
        }
