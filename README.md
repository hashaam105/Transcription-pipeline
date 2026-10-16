# Audio Transcription Pipeline

A small, production-minded service that turns audio (WAV, MP3, M4A, FLAC, OGG, WebM) into
clean, timestamped text for downstream use. It is exposed both as a REST API and a CLI.

```
Audio upload → Validate → Normalize (16 kHz mono WAV) → Chunk on silence → Speech-to-text (with retry)
            → Merge + fix timestamps → Clean text → JSON / SRT / VTT / TXT
```

## Features

- **Many input formats.** Everything is decoded with ffmpeg (via pydub) and normalized to 16 kHz mono 16-bit PCM, so the STT stage never deals with formats.
- **Speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).** Open source and offline. It uses the GPU (float16) when one is available and falls back to CPU (int8). VAD skips silence, and the language is detected automatically.
- **Pluggable engines.** Engines implement the `Transcriber` interface. A deterministic `MockTranscriber` lets the pipeline and tests run without a model.
- **Timestamps for each segment** (seconds plus `HH:MM:SS.mmm`), optional word-level timestamps, and a confidence score for each segment.
- **Long audio.** The audio is split into ~60 s chunks, cut at the quietest point near each boundary so words aren't split. Each chunk's timestamps are shifted to absolute time. The merged segments are sorted and re-indexed, and where two segments overlap, words already transcribed are removed (by word timestamps when available, otherwise by matching the repeated words).
- **Retries.** Transient failures are retried with exponential backoff. Input errors (corrupt or unsupported files) fail fast and are never retried.
- **Async API.** Upload returns `202` with a `job_id` straight away. You can then check status and progress, fetch the result, or retry the job. A synchronous endpoint is available for short clips; it runs the model in a worker thread so the event loop is never blocked.
- **Jobs survive restarts.** Jobs and results are stored in SQLite. On startup, jobs that were queued or mid-processing are re-queued automatically.
- **Downstream-ready output.** You get the cleaned text (fillers removed) alongside the raw text, plus word count, average confidence, and a list of low-confidence segments to flag for human review.

## Project layout

```
transcription_service/
  config.py        # env-driven settings
  audio.py         # validation, normalization, silence-aware chunking
  transcriber.py   # Transcriber interface, Whisper + Mock engines, timestamp utils
  pipeline.py      # orchestration, retry, chunk merge
  postprocess.py   # text clean-up
  formatters.py    # SRT / VTT / TXT output
  api.py           # FastAPI service + SQLite job store + worker pool
  cli.py           # command-line entry point
tests/             # pytest suite (runs on the mock engine)
```

## Setup

Requires Python 3.10–3.12 and [ffmpeg](https://ffmpeg.org/download.html) on `PATH` (needed for MP3/M4A/etc.; WAV works without it).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Usage

### CLI

```bash
python -m transcription_service.cli meeting.mp3                       # JSON to stdout
python -m transcription_service.cli meeting.mp3 --format srt --output meeting.srt
python -m transcription_service.cli meeting.mp3 --language en --word-timestamps
```

### API

```bash
uvicorn transcription_service.api:app --reload
# Interactive docs: http://localhost:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/transcriptions` | Upload audio, returns `202` + `job_id` |
| `POST` | `/v1/transcriptions/sync` | Transcribe a short clip and return the result immediately |
| `GET` | `/v1/transcriptions` | List jobs |
| `GET` | `/v1/transcriptions/{id}` | Status and progress (`queued` / `processing` / `completed` / `failed`) |
| `GET` | `/v1/transcriptions/{id}/result?format=json\|srt\|vtt\|txt` | Transcript |
| `POST` | `/v1/transcriptions/{id}/retry` | Retry a failed job |
| `DELETE` | `/v1/transcriptions/{id}` | Delete the job and its audio |
| `GET` | `/health` | Health check |

```bash
curl -F "file=@meeting.mp3" "http://localhost:8000/v1/transcriptions?word_timestamps=true"
curl http://localhost:8000/v1/transcriptions/<job_id>
curl "http://localhost:8000/v1/transcriptions/<job_id>/result?format=srt"
```

Example result (truncated):

```json
{
  "job_id": "6ac5e11a-...",
  "status": "completed",
  "metadata": { "original_format": "mp3", "original_sample_rate": 44100,
                "original_channels": 2, "duration_sec": 9.0, "chunks": 1 },
  "transcription": {
    "text": "Hello and welcome to the weekly project meeting today we will review ...",
    "raw_text": "um hello and welcome to the weekly project meeting today we will uh review ...",
    "language": "en",
    "word_count": 21,
    "avg_confidence": 0.9,
    "low_confidence_segments": [],
    "segments": [
      { "id": 0, "start": 0.0, "end": 4.0, "start_ts": "00:00:00.000", "end_ts": "00:00:04.000",
        "text": "um hello and welcome to the weekly project meeting", "confidence": 0.9 }
    ],
    "engine": "faster-whisper:base"
  },
  "processing_time_sec": 1.4
}
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `STT_ENGINE` | `whisper` | `whisper` or `mock` |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large-v3`: trades accuracy against speed |
| `MAX_FILE_SIZE_MB` | `25` | Upload limit |
| `MAX_DURATION_SEC` | `3600` | Audio length limit |
| `CHUNK_LENGTH_SEC` | `60` | Target chunk size for long audio |
| `MAX_RETRIES` | `3` | Total attempts per chunk for transient errors |
| `RETRY_BASE_DELAY_SEC` | `2` | Backoff base: waits 2s, then 6s between the 3 attempts |
| `WORKER_THREADS` | `2` | Parallel transcription jobs |
| `MAX_QUEUED_JOBS` | `50` | Above this, uploads get `429` + `Retry-After` |
| `JOBS_DB_PATH` | `storage/jobs.db` | SQLite file for jobs and results |
| `LOW_CONFIDENCE_THRESHOLD` | `0.6` | Segments below this are listed in `low_confidence_segments` |

### Docker

```bash
docker build -t transcription-service .
docker run -p 8000:8000 transcription-service
```

### Tests

```bash
pytest -q
```

The tests use the mock engine and generated audio, so no model download or GPU is needed. They cover
format normalization, rejection of invalid/corrupt/empty input, silence-aware chunking with
absolute timestamps, word timestamps, retry behaviour, output formats, and the async API flow
including concurrent uploads.

## Design decisions

| Decision | Why |
|---|---|
| faster-whisper rather than a cloud API | Free, runs offline (data stays private), strong multilingual accuracy, ~4x faster than reference Whisper. The trade-off is that it needs compute. |
| Normalize everything to 16 kHz mono WAV at the start | Only `audio.py` knows about formats. Any engine gets the same input. |
| `Transcriber` interface + factory | Engines can be swapped (Vosk, cloud APIs) and mocked for tests. |
| Model loaded once at startup | Loading the model is the main latency cost, so it shouldn't happen per request. |
| Chunks cut at the quietest point | Keeps words whole at boundaries and limits memory use for long files. |
| Validation errors separate from transient errors | Corrupt input fails fast with a clear message. Only transient errors are retried. |
| Async jobs + bounded worker pool + 429 backpressure | Uploads stay fast, bursts are absorbed, and the server is never overloaded. |
| Jobs persisted in SQLite, recovered on startup | A restart or crash doesn't lose jobs; unfinished work is picked up again. |
| Clean text alongside raw text | Downstream NLP gets tidy text without losing the original transcript. |

## Taking it to production

This repo keeps the infrastructure in-process (SQLite, a thread pool) so it is easy to run. To scale it, each piece maps to a managed component:

- **Uploads:** presigned S3/GCS URLs with multipart/resumable upload, so files don't pass through the API servers.
- **Storage:** audio in object storage under `audio/{job_id}`, with lifecycle rules moving it to cold storage or deleting it. Jobs and transcripts (segments as JSONB) go in PostgreSQL, with optional Elasticsearch for full-text search. Everything is encrypted, with access per owner and a retention policy (GDPR).
- **Concurrency:** stateless API instances behind a load balancer. The `ThreadPoolExecutor` is replaced by a message queue (SQS / RabbitMQ / Celery+Redis) with GPU workers that autoscale on queue depth, priority queues so long files don't block short ones, and rate limits per API key.
- **Recovery:** at-least-once delivery with idempotent workers, saving each chunk's result as it finishes so only failed chunks are re-run, a dead-letter queue for jobs that keep failing, and a fallback to a smaller model or CPU when the GPU runs out of memory.
- **API:** API-key auth, HTTPS, webhooks when jobs complete, and versioned routes (`/v1`).
- **Observability:** queue depth, latency, error rate and confidence distribution, plus WER tracked against a labelled evaluation set.
