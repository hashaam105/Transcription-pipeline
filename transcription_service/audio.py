"""Audio validation, normalization and chunking."""
from pathlib import Path

from pydub import AudioSegment

from .config import ALLOWED_EXTENSIONS, TARGET_SAMPLE_RATE, settings


class AudioValidationError(Exception):
    """Permanent input error (bad format, corrupt, too large). Never retried."""


def validate_file(path: Path) -> None:
    if not path.exists():
        raise AudioValidationError(f"File not found: {path}")
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported format '{path.suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb == 0:
        raise AudioValidationError("File is empty")
    if size_mb > settings.max_file_size_mb:
        raise AudioValidationError(f"File too large ({size_mb:.1f} MB > {settings.max_file_size_mb} MB)")


def load_and_normalize(path: Path) -> tuple[AudioSegment, dict]:
    """Decode any supported format (via ffmpeg) and normalize to 16 kHz mono 16-bit PCM."""
    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        raise AudioValidationError(f"Could not decode audio (corrupt or unsupported codec): {e}")

    duration_sec = len(audio) / 1000
    if duration_sec == 0:
        raise AudioValidationError("Audio has zero duration")
    if duration_sec > settings.max_duration_sec:
        raise AudioValidationError(f"Audio too long ({duration_sec:.0f}s > {settings.max_duration_sec}s)")

    metadata = {
        "original_format": path.suffix.lower().lstrip("."),
        "original_sample_rate": audio.frame_rate,
        "original_channels": audio.channels,
        "duration_sec": round(duration_sec, 2),
    }
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return audio, metadata


def split_into_chunks(audio: AudioSegment, chunk_sec: int, search_sec: int,
                      window_ms: int = 100) -> list[tuple[float, AudioSegment]]:
    """Split audio into ~chunk_sec pieces, cutting at the quietest point near each boundary
    so words are not split in half. Returns (offset_sec, chunk) pairs."""
    chunk_ms, search_ms, total_ms = chunk_sec * 1000, search_sec * 1000, len(audio)
    chunks, start = [], 0
    while start < total_ms:
        end = min(start + chunk_ms, total_ms)
        if end < total_ms:
            search_start = max(end - search_ms, start + 1)
            best_cut, best_db = end, float("inf")
            for t in range(search_start, end - window_ms + 1, window_ms):
                db = audio[t:t + window_ms].dBFS
                if db < best_db:
                    best_cut, best_db = t + window_ms // 2, db
            end = best_cut
        chunks.append((start / 1000, audio[start:end]))
        start = end
    return chunks
