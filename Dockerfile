# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the current directory into the container at /app
COPY . /app

# Install system dependencies required for uvicorn and cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Include stacking_clf.pkl in the Docker image
COPY stacking_clf.pkl /app

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
