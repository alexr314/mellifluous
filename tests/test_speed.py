"""Tests for the speed= knob: phase-vocoder time-stretch + Reader plumbing.

The vocoder is verified on synthetic sine waves where the expected length
and dominant frequency are known analytically. The Reader integration is
verified with a fake backend that emits known PCM, so we can assert the
post-stretch chunks have the expected lengths without needing a real TTS.
"""
from __future__ import annotations
import warnings

import numpy as np
import pytest

from mellifluous.synthesize._timestretch import time_stretch
from mellifluous.synthesize.types import AudioChunk


# --- phase vocoder ----------------------------------------------------------

SR = 24000


def _tone(freq_hz: float, duration_s: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _peak_freq(pcm: np.ndarray, sr: int = SR) -> float:
    spec = np.abs(np.fft.rfft(pcm))
    bin_idx = int(np.argmax(spec))
    return bin_idx * sr / len(pcm)


class TestTimeStretch:
    def test_speed_one_is_passthrough(self):
        sig = _tone(440, 1.0)
        out = time_stretch(sig, 1.0)
        assert out is sig  # identity, not a copy

    def test_faster_shortens_signal(self):
        sig = _tone(440, 1.0)
        out = time_stretch(sig, 2.0)
        # Stretched at 2x: half the duration.
        assert abs(len(out) - len(sig) // 2) <= 2

    def test_slower_lengthens_signal(self):
        sig = _tone(440, 1.0)
        out = time_stretch(sig, 0.5)
        assert abs(len(out) - len(sig) * 2) <= 2

    @pytest.mark.parametrize("speed", [0.5, 0.75, 1.25, 1.5, 2.0])
    def test_pitch_preserved_across_speeds(self, speed):
        """A 440 Hz tone stays at 440 Hz regardless of speed -- the whole
        point of a phase vocoder vs naive resampling."""
        sig = _tone(440, 1.0)
        out = time_stretch(sig, speed)
        # 1 Hz tolerance (sub-bin) across the tested range.
        assert abs(_peak_freq(out) - 440.0) < 1.0

    def test_empty_input_returns_empty(self):
        out = time_stretch(np.zeros(0, dtype=np.float32), 1.5)
        assert len(out) == 0

    def test_zero_speed_raises(self):
        with pytest.raises(ValueError):
            time_stretch(_tone(440, 0.1), 0.0)

    def test_very_short_input_does_not_crash(self):
        """A buffer shorter than one analysis frame should pass through
        rather than blow up -- the bridge can hand us tiny silence chunks."""
        out = time_stretch(np.zeros(100, dtype=np.float32), 1.5)
        assert isinstance(out, np.ndarray)


# --- Reader.synthesize speed= integration ----------------------------------

class _FakeChunkBackend:
    """Backend that emits a known-length tone per synthesize() call so we
    can verify the Reader's speed= wrapper produces correctly-stretched
    output without needing a real TTS."""
    sample_rate = SR

    def __init__(self):
        self.voice = None
        self.instructions = None

    def warm(self):
        pass

    def synthesize(self, text, *, voice=None, instructions=None):
        # One-second 440 Hz tone, split into two ~equal chunks so we
        # exercise the batching logic.
        sig = _tone(440, 1.0)
        half = len(sig) // 2
        yield AudioChunk(pcm=sig[:half], sample_rate=SR, is_final=False, meta={})
        yield AudioChunk(pcm=sig[half:], sample_rate=SR, is_final=True, meta={})


@pytest.fixture
def fake_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


class TestReaderSpeedIntegration:
    def test_speed_one_is_no_op(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai")
        r.backend = _FakeChunkBackend()
        chunks = list(r.synthesize("hi", as_markdown=False, speed=1.0))
        total = sum(len(c.pcm) for c in chunks)
        assert total == SR  # original 1 second

    def test_speed_above_one_shortens_total_audio(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai")
        r.backend = _FakeChunkBackend()
        chunks = list(r.synthesize("hi", as_markdown=False, speed=1.5))
        total = sum(len(c.pcm) for c in chunks)
        # 1 second at 1.5x -> ~0.667 seconds. Allow a small rounding margin.
        assert abs(total - int(SR / 1.5)) <= SR // 100

    def test_speed_below_one_lengthens_total_audio(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai")
        r.backend = _FakeChunkBackend()
        chunks = list(r.synthesize("hi", as_markdown=False, speed=0.5))
        total = sum(len(c.pcm) for c in chunks)
        assert abs(total - SR * 2) <= SR // 100

    def test_pitch_preserved_through_reader(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai")
        r.backend = _FakeChunkBackend()
        chunks = list(r.synthesize("hi", as_markdown=False, speed=1.5))
        merged = np.concatenate([c.pcm for c in chunks])
        assert abs(_peak_freq(merged) - 440.0) < 1.0

    def test_warning_outside_recommended_range(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai")
        r.backend = _FakeChunkBackend()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list(r.synthesize("hi", as_markdown=False, speed=5.0))
        assert any("recommended" in str(w.message) for w in caught)

    def test_silence_chunks_are_rescaled(self, fake_openai_key):
        """The bridge inserts silence chunks for inter-utterance pauses.
        Those should shrink by 1/speed when speeding up so the structural
        pauses don't make a fast read feel slow."""
        from mellifluous.reader import _stretch_chunks
        sr = SR
        speech = _tone(440, 0.2)
        chunks_in = [
            AudioChunk(pcm=speech, sample_rate=sr, is_final=False, meta={}),
            AudioChunk(
                pcm=np.zeros(sr // 2, dtype=np.float32),
                sample_rate=sr, is_final=False,
                meta={"silence_ms": 500},
            ),
            AudioChunk(pcm=speech, sample_rate=sr, is_final=True, meta={}),
        ]
        out = list(_stretch_chunks(chunks_in, speed=2.0))
        # 500 ms silence at 2x should be reported as 250 ms in the meta
        # and the PCM length should reflect that.
        silence_chunks = [c for c in out if c.meta.get("silence_ms") is not None]
        assert len(silence_chunks) == 1
        assert silence_chunks[0].meta["silence_ms"] == 250
        assert abs(len(silence_chunks[0].pcm) - sr // 4) <= 2
