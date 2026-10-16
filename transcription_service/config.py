"""Central configuration, driven by environment variables."""
import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # Speech-to-text engine: "whisper" (faster-whisper) or "mock"
    stt_engine: str = field(default_factory=lambda: os.getenv("STT_ENGINE", "whisper").lower())
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))

    # Input limits
    max_file_size_mb: int = field(default_factory=lambda: _env_int("MAX_FILE_SIZE_MB", 25))
    max_duration_sec: int = field(default_factory=lambda: _env_int("MAX_DURATION_SEC", 3600))

    # Long audio handling
    chunk_length_sec: int = field(default_factory=lambda: _env_int("CHUNK_LENGTH_SEC", 60))
    chunk_search_window_sec: int = field(default_factory=lambda: _env_int("CHUNK_SEARCH_WINDOW_SEC", 5))

    # Retry policy for transient failures
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 3))
    retry_base_delay_sec: float = field(default_factory=lambda: _env_float("RETRY_BASE_DELAY_SEC", 2.0))

    # Concurrency for the async API
    worker_threads: int = field(default_factory=lambda: _env_int("WORKER_THREADS", 2))
    max_queued_jobs: int = field(default_factory=lambda: _env_int("MAX_QUEUED_JOBS", 50))


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"}
TARGET_SAMPLE_RATE = 16000

settings = Settings()
