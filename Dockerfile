# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directories for models and data if they don't exist
RUN mkdir -p models data/processed

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["python", "inference/api_server.py"]