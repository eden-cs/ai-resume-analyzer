import fitz # PyMuPDF

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