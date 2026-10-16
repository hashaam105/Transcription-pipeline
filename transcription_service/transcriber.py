"""Speech-to-text engines behind a common interface."""
import logging
import math
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    confidence: float
    start_ts: str = ""
    end_ts: str = ""
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    segments: list[Segment] = field(default_factory=list)
    engine: str = ""
    processing_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def format_ts(seconds: float, sep: str = ".") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _norm_token(token: str) -> str:
    return "".join(c for c in token.lower() if c.isalnum())


def _strip_repeated_prefix(prev_text: str, text: str, max_words: int = 8) -> str:
    """Remove the longest run of words at the start of `text` that repeats the end of `prev_text`."""
    prev_tokens, tokens = prev_text.split(), text.split()
    prev_norm, norm = [_norm_token(t) for t in prev_tokens], [_norm_token(t) for t in tokens]
    for n in range(min(max_words, len(prev_tokens), len(tokens)), 0, -1):
        if prev_norm[-n:] == norm[:n]:
            return " ".join(tokens[n:])
    return text


def _dedupe_overlap(seg: Segment, prev: Segment) -> None:
    """Drop the part of `seg` that was already transcribed in `prev` (they overlap in time)."""
    if seg.words:
        seg.words = [w for w in seg.words if (w.start + w.end) / 2 >= prev.end]
        seg.text = " ".join(w.word for w in seg.words)
        if seg.words:
            seg.start = max(seg.start, seg.words[0].start)
    else:
        seg.text = _strip_repeated_prefix(prev.text, seg.text)


def normalize_segments(segments: list[Segment]) -> list[Segment]:
    """Sort, de-duplicate overlapping speech, drop empty segments, re-index, add readable timestamps."""
    cleaned, prev_end = [], 0.0
    for seg in sorted(segments, key=lambda s: s.start):
        if cleaned and seg.start < prev_end:
            _dedupe_overlap(seg, cleaned[-1])
        seg.start = round(max(seg.start, prev_end), 2)
        seg.end = round(seg.end, 2)
        if seg.end <= seg.start or not seg.text.strip():
            continue
        seg.id = len(cleaned)
        seg.start_ts, seg.end_ts = format_ts(seg.start), format_ts(seg.end)
        cleaned.append(seg)
        prev_end = seg.end
    return cleaned


class Transcriber(ABC):
    name = "base"

    @abstractmethod
    def transcribe(self, wav_path: Path, language: str | None = None,
                   word_timestamps: bool = False) -> TranscriptionResult:
        """Transcribe a 16 kHz mono WAV file. Timestamps are relative to the file start."""


class WhisperTranscriber(Transcriber):
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel

        device = "cuda" if self._cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        log.info("Loading Whisper '%s' on %s (%s)", model_size, device, compute_type)
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.name = f"faster-whisper:{model_size}"

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def transcribe(self, wav_path: Path, language: str | None = None,
                   word_timestamps: bool = False) -> TranscriptionResult:
        start = time.perf_counter()
        segments_iter, info = self.model.transcribe(
            str(wav_path),
            language=language,
            beam_size=5,
            vad_filter=True,  # skip silence, reduce hallucinations
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=word_timestamps,
        )
        segments = [
            Segment(
                id=i, start=s.start, end=s.end, text=s.text.strip(),
                confidence=round(min(1.0, math.exp(s.avg_logprob)), 3),
                words=[Word(round(w.start, 2), round(w.end, 2), w.word.strip(), round(w.probability, 3))
                       for w in (s.words or [])],
            )
            for i, s in enumerate(segments_iter)
        ]
        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            segments=segments,
            engine=self.name,
            processing_time_sec=round(time.perf_counter() - start, 2),
        )


class MockTranscriber(Transcriber):
    """Deterministic engine for tests and environments without a model.
    Emits one segment every 4 seconds of audio."""
    name = "mock"
    SENTENCES = [
        "um hello and welcome to the weekly project meeting",
        "today we will uh review the transcription pipeline",
        "the main goal is to you know improve accuracy",
        "next we will discuss deployment and monitoring",
    ]

    def transcribe(self, wav_path: Path, language: str | None = None,
                   word_timestamps: bool = False) -> TranscriptionResult:
        with wave.open(str(wav_path), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()

        segments, t, i = [], 0.0, 0
        while t < duration:
            end = min(t + 4.0, duration)
            text = self.SENTENCES[i % len(self.SENTENCES)]
            words = []
            if word_timestamps:
                tokens = text.split()
                step = (end - t) / len(tokens)
                words = [Word(round(t + k * step, 2), round(t + (k + 1) * step, 2), w, 0.95)
                         for k, w in enumerate(tokens)]
            segments.append(Segment(id=i, start=round(t, 2), end=round(end, 2), text=text,
                                    confidence=0.9, words=words))
            t, i = end, i + 1

        return TranscriptionResult(
            text=" ".join(s.text for s in segments),
            language=language or "en",
            language_probability=1.0,
            segments=segments,
            engine=self.name,
        )


def get_transcriber(engine: str | None = None) -> Transcriber:
    engine = (engine or settings.stt_engine).lower()
    if engine == "mock":
        return MockTranscriber()
    try:
        return WhisperTranscriber(settings.whisper_model)
    except Exception as e:
        log.warning("Whisper unavailable (%s); falling back to mock engine", e)
        return MockTranscriber()
