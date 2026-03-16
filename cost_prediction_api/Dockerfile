FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY app.py .

# Copy model files
COPY model/ model/

# Hugging Face Spaces runs on port 7860
ENV PORT=7860
ENV MODEL_PATH=model/model.pkl
ENV META_PATH=model/model_meta.json

EXPOSE 7860

# Run with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
