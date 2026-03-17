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


def generate_suggestions(missing: dict, score: float) -> str:
    """
    This helper function generates suggestions for improving the resume based on the missing keywords and the job description using OpenAI's language model.

    @param missing: dict, a dictionary of missing keywords categorized by importance level.

    @param score: float, the match score as a percentage.

    @return: str, the generated suggestions for improving the resume.
    """
    # TODO: check my model isn't global and if REQUEST_COUNT needs to be global

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
    