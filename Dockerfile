# Base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for OpenCV & Tesseract)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of your code: app/, ocr_pipeline/, model/, etc.
COPY . .

# Expose port (optional but recommended for clarity)
EXPOSE 8000

# Default command to run FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
