FROM python:3.x-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y swig libfaiss-dev build-essential

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your app code
COPY . .

# Command to run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]