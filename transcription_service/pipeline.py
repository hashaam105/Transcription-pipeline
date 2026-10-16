"""End-to-end pipeline: validate -> normalize -> chunk -> transcribe (with retry) -> merge -> post-process."""
import logging
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from . import audio as audio_utils
from .audio import AudioValidationError
from .config import settings
from .postprocess import clean_text, word_count
from .transcriber import Segment, Transcriber, Word, get_transcriber, normalize_segments

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def with_retry(fn: Callable, *, retries: int, base_delay: float, label: str):
    """Retry transient failures with exponential backoff. Input errors are never retried.
    `retries` is the total number of attempts, so 3 attempts wait base_delay, then base_delay * 3."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except AudioValidationError:
            raise
        except Exception as e:
            if attempt == retries:
                raise
            delay = base_delay * (3 ** (attempt - 1))
            log.warning("%s failed (attempt %d/%d): %s. Retrying in %.1fs", label, attempt, retries, e, delay)
            time.sleep(delay)


class TranscriptionPipeline:
    def __init__(self, transcriber: Transcriber | None = None):
        self.transcriber = transcriber or get_transcriber()

    def run(self, path: Path, *, language: str | None = None, word_timestamps: bool = False,
            job_id: str | None = None, on_progress: ProgressCallback | None = None) -> dict:
        job_id = job_id or str(uuid.uuid4())
        started = time.perf_counter()
        log.info("Job %s: received %s", job_id, path.name)

        audio_utils.validate_file(path)
        audio, metadata = audio_utils.load_and_normalize(path)
        chunks = audio_utils.split_into_chunks(audio, settings.chunk_length_sec,
                                               settings.chunk_search_window_sec)
        log.info("Job %s: %.1fs audio, %d chunk(s)", job_id, metadata["duration_sec"], len(chunks))

        all_segments: list[Segment] = []
        language_votes: dict[str, float] = {}

        with tempfile.TemporaryDirectory() as tmp:
            for idx, (offset, chunk) in enumerate(chunks):
                chunk_path = Path(tmp) / f"chunk_{idx:04d}.wav"
                chunk.export(chunk_path, format="wav")

                result = with_retry(
                    lambda: self.transcriber.transcribe(chunk_path, language, word_timestamps),
                    retries=settings.max_retries, base_delay=settings.retry_base_delay_sec,
                    label=f"Job {job_id} chunk {idx}",
                )
                language_votes[result.language] = (language_votes.get(result.language, 0)
                                                   + result.language_probability * len(chunk))
                for seg in result.segments:
                    seg.start += offset
                    seg.end += offset
                    seg.words = [Word(round(w.start + offset, 2), round(w.end + offset, 2),
                                      w.word, w.probability) for w in seg.words]
                    all_segments.append(seg)
                chunk_path.unlink(missing_ok=True)
                if on_progress:
                    on_progress(idx + 1, len(chunks))

        segments = normalize_segments(all_segments)
        raw_text = " ".join(s.text for s in segments)
        clean = clean_text(raw_text)
        detected_language = max(language_votes, key=language_votes.get) if language_votes else (language or "unknown")

        return {
            "job_id": job_id,
            "file_name": path.name,
            "status": "completed",
            "metadata": {**metadata, "chunks": len(chunks)},
            "transcription": {
                "text": clean,
                "raw_text": raw_text,
                "language": detected_language,
                "word_count": word_count(clean),
                "avg_confidence": round(sum(s.confidence for s in segments) / len(segments), 3) if segments else 0.0,
                "low_confidence_segments": [s.id for s in segments
                                            if s.confidence < settings.low_confidence_threshold],
                "segments": [_segment_dict(s, word_timestamps) for s in segments],
                "engine": self.transcriber.name,
            },
            "processing_time_sec": round(time.perf_counter() - started, 2),
        }


def _segment_dict(seg: Segment, include_words: bool) -> dict:
    d = {"id": seg.id, "start": seg.start, "end": seg.end, "start_ts": seg.start_ts,
         "end_ts": seg.end_ts, "text": seg.text, "confidence": seg.confidence}
    if include_words:
        d["words"] = [vars(w) for w in seg.words]
    return d
