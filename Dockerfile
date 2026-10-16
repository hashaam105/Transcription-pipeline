FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY transcription_service ./transcription_service

ENV STT_ENGINE=whisper WHISPER_MODEL=base
EXPOSE 8000
CMD ["uvicorn", "transcription_service.api:app", "--host", "0.0.0.0", "--port", "8000"]
