from dataclasses import replace
from pathlib import Path

import pytest

from transcription_service import pipeline as pipeline_module
from transcription_service.audio import AudioValidationError, load_and_normalize, split_into_chunks
from transcription_service.formatters import to_srt, to_vtt
from transcription_service.pipeline import TranscriptionPipeline, with_retry
from transcription_service.postprocess import clean_text
from transcription_service.transcriber import (MockTranscriber, Segment, Word, format_ts,
                                               normalize_segments)


@pytest.fixture
def pipeline():
    return TranscriptionPipeline(MockTranscriber())


def test_normalizes_to_16k_mono(short_wav):
    audio, meta = load_and_normalize(short_wav)
    assert audio.frame_rate == 16000 and audio.channels == 1
    assert meta["original_sample_rate"] == 44100 and meta["original_channels"] == 2
    assert meta["duration_sec"] == pytest.approx(10, abs=0.1)


def test_rejects_unsupported_format(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    with pytest.raises(AudioValidationError, match="Unsupported format"):
        TranscriptionPipeline(MockTranscriber()).run(bad)


def test_rejects_corrupt_audio(tmp_path, pipeline):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not really audio")
    with pytest.raises(AudioValidationError, match="Could not decode"):
        pipeline.run(bad)


def test_rejects_empty_file(tmp_path, pipeline):
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(AudioValidationError, match="empty"):
        pipeline.run(empty)


def test_short_file_transcription(short_wav, pipeline):
    result = pipeline.run(short_wav)
    t = result["transcription"]
    assert result["status"] == "completed"
    assert result["metadata"]["chunks"] == 1
    assert t["segments"][0]["start"] == 0.0
    assert t["segments"][-1]["end"] == pytest.approx(10, abs=0.1)
    assert "um" not in t["text"].lower().split()  # fillers removed from clean text
    assert "um" in t["raw_text"]                    # raw text preserved


def test_long_audio_chunks_on_silence_and_keeps_global_timestamps(long_wav, pipeline):
    result = pipeline.run(long_wav)
    segs = result["transcription"]["segments"]
    assert result["metadata"]["chunks"] == 3
    # timestamps are absolute, sorted and non-overlapping
    for a, b in zip(segs, segs[1:]):
        assert a["end"] <= b["start"]
    assert segs[-1]["end"] == pytest.approx(result["metadata"]["duration_sec"], abs=0.1)
    assert [s["id"] for s in segs] == list(range(len(segs)))


def test_chunk_boundaries_fall_in_silence(long_wav):
    audio, _ = load_and_normalize(long_wav)
    chunks = split_into_chunks(audio, chunk_sec=20, search_sec=5)
    offsets = [o for o, _ in chunks]
    assert 18.0 <= offsets[1] <= 19.0  # first silent gap is 18s-19s
    assert sum(len(c) for _, c in chunks) == len(audio)  # nothing lost


def test_word_timestamps(short_wav, pipeline):
    seg = pipeline.run(short_wav, word_timestamps=True)["transcription"]["segments"][0]
    assert seg["words"] and seg["words"][0]["start"] >= seg["start"]


def test_retry_recovers_from_transient_error():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("GPU out of memory")
        return "ok"

    assert with_retry(flaky, retries=3, base_delay=0, label="test") == "ok"
    assert calls["n"] == 3


def test_retry_does_not_retry_validation_errors():
    calls = {"n": 0}

    def bad_input():
        calls["n"] += 1
        raise AudioValidationError("corrupt")

    with pytest.raises(AudioValidationError):
        with_retry(bad_input, retries=3, base_delay=0, label="test")
    assert calls["n"] == 1


def test_formatters(short_wav, pipeline):
    t = pipeline.run(short_wav)["transcription"]
    assert to_srt(t).startswith("1\n00:00:00,000 --> 00:00:04,000")
    assert to_vtt(t).startswith("WEBVTT")


def test_format_ts():
    assert format_ts(3725.5) == "01:02:05.500"
    assert format_ts(2.5, ",") == "00:00:02,500"


def test_clean_text():
    assert clean_text("um hello uh  world") == "Hello world."


def test_overlapping_segments_drop_repeated_words():
    segs = normalize_segments([
        Segment(id=0, start=0.0, end=5.0, text="we will review the pipeline", confidence=0.9),
        Segment(id=1, start=4.0, end=8.0, text="the pipeline today", confidence=0.9),
    ])
    assert [s.text for s in segs] == ["we will review the pipeline", "today"]
    assert segs[1].start == 5.0


def test_overlapping_segments_use_word_timestamps():
    segs = normalize_segments([
        Segment(id=0, start=0.0, end=2.0, text="hello world", confidence=0.9,
                words=[Word(0.0, 1.0, "hello", 0.9), Word(1.0, 2.0, "world", 0.9)]),
        Segment(id=1, start=1.2, end=3.0, text="world again", confidence=0.9,
                words=[Word(1.2, 1.8, "world", 0.9), Word(2.0, 3.0, "again", 0.9)]),
    ])
    assert segs[1].text == "again" and segs[1].start == 2.0


def test_non_overlapping_repetition_is_kept():
    segs = normalize_segments([
        Segment(id=0, start=0.0, end=2.0, text="no no", confidence=0.9),
        Segment(id=1, start=2.0, end=4.0, text="no way", confidence=0.9),
    ])
    assert segs[1].text == "no way"


def test_low_confidence_threshold_is_configurable(short_wav, pipeline, monkeypatch):
    monkeypatch.setattr(pipeline_module, "settings", replace(pipeline_module.settings, low_confidence_threshold=0.95))
    t = pipeline.run(short_wav)["transcription"]
    assert t["low_confidence_segments"] == [s["id"] for s in t["segments"]]  # mock confidence is 0.9
