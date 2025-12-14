# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Default command: run FastAPI server
# You can also train manually inside container if needed
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
