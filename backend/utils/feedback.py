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