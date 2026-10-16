import os

# Tests run with the mock engine and no retry delay: no model download or GPU needed.
os.environ["STT_ENGINE"] = "mock"
os.environ["RETRY_BASE_DELAY_SEC"] = "0"
os.environ["CHUNK_LENGTH_SEC"] = "20"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from pydub import AudioSegment  # noqa: E402
from pydub.generators import Sine  # noqa: E402


def make_wav(path: Path, seconds: float, sample_rate: int = 44100, channels: int = 2) -> Path:
    tone = Sine(440).to_audio_segment(duration=int(seconds * 1000)).apply_gain(-10)
    tone.set_frame_rate(sample_rate).set_channels(channels).export(path, format="wav")
    return path


@pytest.fixture
def short_wav(tmp_path) -> Path:
    return make_wav(tmp_path / "short.wav", 10)


@pytest.fixture
def long_wav(tmp_path) -> Path:
    # Tone with a silent gap near each 20s boundary, so the chunker can cut on silence
    tone = Sine(440).to_audio_segment(duration=18_000).apply_gain(-10)
    gap = AudioSegment.silent(duration=1_000)
    audio = tone + gap + tone + gap + tone
    path = tmp_path / "long.wav"
    audio.set_frame_rate(44100).export(path, format="wav")
    return path
