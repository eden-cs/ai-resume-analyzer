from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz # PyMuPDF

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

@app.post("/analyze")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    # Converts UploadFile to raw bytes
    file_bytes = await file.read()

    # Calls on helper function to extract the text from the PDF
    resume_text = extract_pdf_text(file_bytes)

    return {
        "message": "Resume received", 
        "resume_length": len(resume_text),
        "job_desc": job_desc
        }

@app.get("/")
def root():
    return {
        "status": "API is running"
        }
