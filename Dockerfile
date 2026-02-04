FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY colors/ ./colors/

RUN mkdir -p /data/attachments

ENV DATABASE_PATH=/data/emails.db \
    ATTACHMENTS_PATH=/data/attachments \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD uvicorn main:app --host $HOST --port $PORT