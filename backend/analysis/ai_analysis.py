import json
import os
from google import genai
from google.genai import types
import google.api_core.exceptions as google_exceptions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Google Generative AI client with API key from the environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Rate limiting - max requests per day
REQUEST_COUNT = 0
MAX_DAILY_REQUESTS = 20

# Return feedback based on missing keywords if API call fails
def _fallback_suggestions(missing: dict) -> list:
    """
    This helper function returns hardcoded fallback suggestions when the Gemini API is unavailable.

    @param missing: dict, a dictionary of missing keywords categorized by importance level.

    @return: list, a list of fallback suggestions for improving the resume.
    """
    
    high_missing = ", ".join(missing["high"]) if missing["high"] else "none"
    medium_missing = ", ".join(missing["medium"]) if missing["medium"] else "none"

    return [
        f"Focus on adding these missing high importance skills to your resume: {high_missing}.",
        f"Consider including these medium importance skills: {medium_missing}.", 
        "Tailor your resume to better match the job description language.",
        "Quantify your achievements with specific metrics or examples to demonstrate your impact." 
    ]


def generate_suggestions(missing: dict, score: float) -> list:
    """
    This helper function generates suggestions for improving the resume based on the missing keywords and the job description using OpenAI's language model.

    @param missing: dict, a dictionary of missing keywords categorized by importance level.

    @param score: float, the match score as a percentage.

    @return: list, the generated suggestions for improving the resume.
    """
    global REQUEST_COUNT
    
    # Check if the daily request limit has been reached
    if (REQUEST_COUNT >= MAX_DAILY_REQUESTS):
        return _fallback_suggestions(missing)
    
    REQUEST_COUNT += 1
    
    # Format missing keywords for the prompt
    high_missing = ", ".join(missing["high"]) if missing["high"] else "none"
    medium_missing = ", ".join(missing["medium"]) if missing["medium"] else "none"
    
    # Create a prompt for the language model based on the missing keywords and match score
    prompt = f"""You are a career coach helping a student improve their resume for a specific job application.

    The resume has a match score of {score}% against the job description.

    High importance keywords missing: {high_missing}
    Medium importance keywords missing: {medium_missing}

    Generate exactly 4 specific, actionable suggestions. For each missing high importance keyword:
    - Suggest exactly HOW to add it to the resume naturally
    - If it is a technical skill, suggest a specific free resource or project to build experience with it

    Be specific and direct. Instead of "add keywords to your resume" say "Add a bullet point under your X project mentioning Y".
    Instead of "demonstrate familiarity" say "Take the free Google course on X at coursera.org".

    Return ONLY a valid JSON array of exactly 4 strings. No preamble, no markdown, no bullet points inside the strings. Format:
    ["specific suggestion 1", "specific suggestion 2", "specific suggestion 3", "specific suggestion 4"]"""

    # Try to generate suggestions using the language model with error handling for potential API errors
    try:
        response = client.models.generate_content (
            model = "gemini-2.5-flash",
            contents = prompt, 
            config = types.GenerateContentConfig (
                max_output_tokens = 1024, # Limit the response to 1024 tokens to ensure concise suggestions
                temperature = 0.7, # Set temperature to 0.7 for a balance between creativity and relevance
                top_p = 0.9 # Set top_p to 0.9 to consider the top 90% of token probabilities for generating suggestions
            )
        )

        # Generate suggestions using the language model
        suggestions_text = response.text.strip() if response and response.text else "[]"
    
        # Remove markdown code blocks if Gemini wrapped the JSON in them.
        suggestions_text = suggestions_text.replace("```json", "").replace("```", "").strip()

        # If response is empty or too short to be a valid JSON, then fall back immediately.
        if len(suggestions_text) < 10:
            print("Gemini returned an unexpectedly short response or returned no suggestions.")
            return _fallback_suggestions(missing)
        
        # Conver JSON string into Python list
        suggestions = json.loads(suggestions_text)

        # Validate response format
        if not isinstance(suggestions, list):
            raise ValueError("Gemini response is not a list.")
        
        # Ensure all items are strings
        suggestions = [str(suggestion) for suggestion in suggestions]

        # Ensure there are exactly 4 suggestions.
        suggestions = suggestions[:4]

        # Fallback if Gemini returned an empty list
        if len(suggestions) == 0:
            raise ValueError("Gemini returned empty suggestions.")
        
        return suggestions
    
    except json.JSONDecodeError as e:
        print("JSON parsing failed:", str(e))
        
    except google_exceptions.InvalidArgument as e:
        print("Invalid request:", str(e))

    except google_exceptions.ResourceExhausted as e:
        print("Quota exceeded or rate limit hit. Please try again tomorrow:", str(e))

    except google_exceptions.GoogleAPIError as e:
        print("API call failed:", str(e))

    except Exception as e:
        print("An unexpected error occurred:", str(e))
        error_str = str(e)
        # Handle quota errors caught by generic exception handler
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            print("Quota exceeded:", error_str[:100])
        else:
            print("An unexpected error occurred:", error_str[:100])

    return _fallback_suggestions(missing)    