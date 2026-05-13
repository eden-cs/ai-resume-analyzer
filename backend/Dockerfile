# Start from the offical Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory inside container
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy medium model
RUN python -m spacy download en_core_web_md

# Copy the rest of the backend code into the container
COPY . .

# Document that the container listens on port 8000
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]