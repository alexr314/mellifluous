"""Sinks: functions that consume an iterator of AudioChunk."""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import soundfile as sf

from .types import AudioChunk


def collect(chunks: Iterable[AudioChunk]) -> tuple[np.ndarray, int]:
    """Drain to a single (pcm, sample_rate)."""
    pieces, sr = [], None
    for c in chunks:
        if sr is None: sr = c.sample_rate
        pieces.append(c.pcm)
    if not pieces:
        return np.zeros(0, dtype=np.float32), sr or 24000
    return np.concatenate(pieces), sr


def write_wav(path: str | Path, chunks: Iterable[AudioChunk], *, subtype: str = "PCM_16") -> Path:
    """Drain `chunks` and write a WAV file. Returns the path."""
    path = Path(path)
    pcm, sr = collect(chunks)
    sf.write(str(path), pcm, sr, subtype=subtype)
    return path


def play(
    chunks: Iterable[AudioChunk],
    *,
    blocking: bool = True,
    prebuffer_ms: int = 200,
    block_ms: int = 20,
    tail_pad_ms: int = 600,
) -> None:
    """Play chunks on the default output device as they arrive.

    Callback-driven OutputStream pulls from a ring buffer that the producer
    appends to. This decouples the model-generation thread from the audio
    thread, so slow chunks don't underrun the device.

    `tail_pad_ms` of silence is appended to the very end so the OS audio
    pipeline has time to flush before we stop the stream. Without this the
    last few hundred milliseconds can be cut off.
    """
    import sounddevice as sd
    import time as _time

    sample_rate: Optional[int] = None
    buffer = np.zeros(0, dtype=np.float32)
    buffer_lock = threading.Lock()
    # `drain_target` is the total number of samples we have pushed in. The
    # audio callback advances `played_samples` as it consumes them. When
    # `played_samples >= drain_target` AND the producer has finished, we
    # know everything has been handed to the device.
    pushed_samples = 0
    played_samples = 0
    producer_done = threading.Event()
    drained = threading.Event()

    def audio_cb(outdata, frames, time_info, status):
        nonlocal buffer, played_samples
        with buffer_lock:
            n = min(frames, len(buffer))
            if n > 0:
                outdata[:n, 0] = buffer[:n]
                buffer = buffer[n:]
                played_samples += n
            if n < frames:
                outdata[n:, 0] = 0.0
            if producer_done.is_set() and played_samples >= pushed_samples:
                drained.set()

    stream: Optional["sd.OutputStream"] = None
    started = False
    prebuffer_samples = 0

    try:
        for c in chunks:
            if sample_rate is None:
                sample_rate = c.sample_rate
                prebuffer_samples = int(prebuffer_ms / 1000 * sample_rate)
                blocksize = max(1, int(block_ms / 1000 * sample_rate))
                stream = sd.OutputStream(
                    samplerate=sample_rate, channels=1, dtype="float32",
                    blocksize=blocksize, callback=audio_cb,
                )

            pcm = np.ascontiguousarray(c.pcm, dtype=np.float32)
            with buffer_lock:
                buffer = np.concatenate((buffer, pcm))
                pushed_samples += len(pcm)

            if not started:
                with buffer_lock:
                    have = len(buffer)
                if have >= prebuffer_samples:
                    stream.start()
                    started = True

        # Append a tail of silence so the device finishes the last real audio
        # before we tear down the stream.
        if stream is not None and sample_rate is not None and tail_pad_ms > 0:
            tail = np.zeros(int(tail_pad_ms / 1000 * sample_rate), dtype=np.float32)
            with buffer_lock:
                buffer = np.concatenate((buffer, tail))
                pushed_samples += len(tail)

        if not started and stream is not None:
            stream.start()
            started = True
        producer_done.set()
    finally:
        if stream is not None:
            if blocking and started:
                # Wait until every buffered sample has been delivered to the
                # device. No timeout: when the producer outpaces playback
                # (TTS RTF < 1), the buffer at producer-end can hold tens of
                # seconds of audio that still needs to play. A bounded wait
                # here would silently cut the tail off.
                drained.wait()
                # The `drained` event fires when the audio thread has consumed
                # all bytes from our buffer, but those bytes are still inside
                # the OS audio pipeline (CoreAudio buffer + DAC + speakers).
                # We sleep long enough for that pipeline to flush before we
                # stop the stream. sd.OutputStream.latency is typically ~10ms;
                # we add a comfortable margin so the tail of the last word
                # actually reaches the speakers.
                _time.sleep(max(stream.latency or 0.0, 0.05) + 0.25)
            stream.stop()
            stream.close()
