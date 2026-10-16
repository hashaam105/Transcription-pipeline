"""REST API.

POST /v1/transcriptions            async upload -> 202 + job_id
POST /v1/transcriptions/sync       sync transcription for short clips
GET  /v1/transcriptions            list jobs
GET  /v1/transcriptions/{id}       job status / progress
GET  /v1/transcriptions/{id}/result?format=json|srt|vtt|txt
POST /v1/transcriptions/{id}/retry retry a failed job
DELETE /v1/transcriptions/{id}     delete job and stored audio
GET  /health
"""
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .audio import AudioValidationError
from .config import ALLOWED_EXTENSIONS, settings
from .formatters import FORMATTERS
from .pipeline import TranscriptionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STORAGE_DIR = Path("storage/uploads")  # stand-in for object storage (S3/GCS)
UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(title="Transcription Service", version="1.0.0")
pipeline = TranscriptionPipeline()  # model loaded once at startup
executor = ThreadPoolExecutor(max_workers=settings.worker_threads)  # stand-in for a job queue + workers


class JobStore:
    """Thread-safe in-memory job store. In production: PostgreSQL / Redis."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, job: dict) -> None:
        with self._lock:
            self._jobs[job["job_id"]] = job

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields, updated_at=_now())

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def delete(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.pop(job_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(j["status"] in ("queued", "processing") for j in self._jobs.values())


jobs = JobStore()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _save_upload(file: UploadFile, job_id: str) -> Path:
    """Stream the upload to disk in chunks, enforcing the size limit without loading it into memory."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    job_dir = STORAGE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"input{suffix}"
    max_bytes, written = settings.max_file_size_mb * 1024 * 1024, 0
    with dest.open("wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"File too large. Max {settings.max_file_size_mb} MB.")
            out.write(chunk)
    if written == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "File is empty")
    return dest


def _run_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    jobs.update(job_id, status="processing", progress=0, error=None)
    try:
        result = pipeline.run(
            Path(job["audio_path"]), language=job["language"], word_timestamps=job["word_timestamps"],
            job_id=job_id,
            on_progress=lambda done, total: jobs.update(job_id, progress=int(done / total * 100)),
        )
        result["file_name"] = job["file_name"]
        jobs.update(job_id, status="completed", progress=100, result=result)
    except AudioValidationError as e:
        jobs.update(job_id, status="failed", error=str(e), retryable=False)
    except Exception as e:
        log.exception("Job %s failed", job_id)
        jobs.update(job_id, status="failed", error=f"Processing error: {e}", retryable=True)


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("job_id", "file_name", "status", "progress", "error",
                                    "retryable", "created_at", "updated_at")}


@app.post("/v1/transcriptions", status_code=202)
async def create_transcription(
    file: UploadFile = File(...),
    language: str | None = Query(None, description="ISO code, e.g. 'en'. Empty = auto-detect"),
    word_timestamps: bool = Query(False),
):
    if jobs.active_count() >= settings.max_queued_jobs:
        return JSONResponse({"error": "Too many jobs in progress, try again later"}, status_code=429,
                            headers={"Retry-After": "30"})
    job_id = str(uuid.uuid4())
    audio_path = await _save_upload(file, job_id)
    jobs.create({
        "job_id": job_id, "file_name": file.filename, "audio_path": str(audio_path),
        "language": language, "word_timestamps": word_timestamps,
        "status": "queued", "progress": 0, "error": None, "retryable": None,
        "result": None, "created_at": _now(), "updated_at": _now(),
    })
    executor.submit(_run_job, job_id)
    return {"job_id": job_id, "status": "queued", "status_url": f"/v1/transcriptions/{job_id}"}


@app.post("/v1/transcriptions/sync")
async def transcribe_sync(
    file: UploadFile = File(...),
    language: str | None = Query(None),
    word_timestamps: bool = Query(False),
    format: Literal["json", "srt", "vtt", "txt"] = Query("json"),
):
    job_id = str(uuid.uuid4())
    audio_path = await _save_upload(file, job_id)
    try:
        result = pipeline.run(audio_path, language=language, word_timestamps=word_timestamps, job_id=job_id)
        result["file_name"] = file.filename
    except AudioValidationError as e:
        raise HTTPException(400, str(e))
    except Exception:
        log.exception("Sync transcription failed")
        raise HTTPException(500, "Internal processing error")
    finally:
        shutil.rmtree(audio_path.parent, ignore_errors=True)
    return _render(result, format)


@app.get("/v1/transcriptions")
def list_transcriptions():
    return [_public(j) for j in jobs.list()]


@app.get("/v1/transcriptions/{job_id}")
def get_transcription(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return _public(job)


@app.get("/v1/transcriptions/{job_id}/result")
def get_result(job_id: str, format: Literal["json", "srt", "vtt", "txt"] = Query("json")):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job is {job['status']}" + (f": {job['error']}" if job["error"] else ""))
    return _render(job["result"], format)


@app.post("/v1/transcriptions/{job_id}/retry", status_code=202)
def retry_transcription(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "failed":
        raise HTTPException(409, f"Only failed jobs can be retried (current: {job['status']})")
    jobs.update(job_id, status="queued", progress=0, error=None)
    executor.submit(_run_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.delete("/v1/transcriptions/{job_id}", status_code=204)
def delete_transcription(job_id: str):
    job = jobs.delete(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    shutil.rmtree(Path(job["audio_path"]).parent, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok", "engine": pipeline.transcriber.name, "active_jobs": jobs.active_count()}


def _render(result: dict, fmt: str):
    if fmt == "json":
        return result
    formatter, media_type = FORMATTERS[fmt]
    return PlainTextResponse(formatter(result["transcription"]), media_type=media_type)
