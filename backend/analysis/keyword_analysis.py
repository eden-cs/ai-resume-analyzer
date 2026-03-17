import spacy # NLP library
from collections import Counter

# Load the small English NLP model from spaCy
nlp = spacy.load("en_core_web_sm")

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