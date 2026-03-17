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
