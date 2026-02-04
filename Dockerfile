FROM python:3.11-slim

# Install minimal system dependencies
# Removed GUI libs since we're using opencv-python-headless
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
# Tip: Ensure opencv-python-headless is in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY colors/ ./colors/

# Create directories for persistent data (Railway Volumes should map here)
RUN mkdir -p /data/attachments

# Set environment variables
ENV DATABASE_PATH=/data/emails.db \
    ATTACHMENTS_PATH=/data/attachments \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

# Use the shell form or env var to ensure Railway's dynamic port is respected
CMD uvicorn main:app --host $HOST --port $PORT