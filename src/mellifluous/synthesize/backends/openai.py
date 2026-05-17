"""OpenAI cloud TTS backend.

Streams 24 kHz mono PCM from the OpenAI audio API. Supported models:
  - "gpt-4o-mini-tts"  (default — cheap, conversational, supports `instructions`)
  - "tts-1"
  - "tts-1-hd"

Voice is a string from OpenAI's preset list (no cloning). `instructions`
steers tone for `gpt-4o-mini-tts` only and is silently ignored by older
`tts-1*` models.

Requires the `openai` SDK (added as a required dependency in pyproject.toml).
"""
from __future__ import annotations
import logging
import os
import time
from typing import Iterable, Iterator, Optional, Union

import numpy as np

from ..base   import Backend
from ..types  import AudioChunk
from ..voices import Voice
from .._text  import drain_sentences

log = logging.getLogger(__name__)


MODELS = ("gpt-4o-mini-tts", "tts-1", "tts-1-hd")
DEFAULT_MODEL = "gpt-4o-mini-tts"

# OpenAI's documented preset voices as of May 2026.
VOICES = (
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "nova", "onyx", "sage", "shimmer", "verse",
)
DEFAULT_VOICE = "ash"

# OpenAI's pcm response format is 24 kHz mono signed 16-bit little-endian.
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2

# Stream pcm in ~100 ms chunks so playback can start with low latency. Smaller
# chunks add HTTP-overhead-per-yield without audible benefit; larger chunks
# delay first-audio.
_STREAM_CHUNK_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE // 10  # 4800 bytes = 100ms

# `instructions` is only honored by gpt-4o-mini-tts.
_MODELS_WITH_INSTRUCTIONS = {"gpt-4o-mini-tts"}


def _resolve_voice(voice: Optional[Union[str, Voice]]) -> str:
    if voice is None:
        return DEFAULT_VOICE
    if isinstance(voice, str):
        if voice not in VOICES:
            raise ValueError(
                f"unknown openai voice {voice!r}. choices: {list(VOICES)}"
            )
        return voice
    raise TypeError(
        f"openai backend expects voice as a string preset name "
        f"(one of {list(VOICES)}); got {type(voice).__name__}"
    )


class OpenAIBackend(Backend):
    """OpenAI cloud TTS (gpt-4o-mini-tts / tts-1 / tts-1-hd)."""

    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}. choices: {list(MODELS)}")
        self.model = model
        self.voice = _resolve_voice(voice)
        self.instructions = instructions
        # OpenAI SDK reads OPENAI_API_KEY from the env automatically when
        # api_key is None, but we resolve explicitly so a missing key fails
        # at construction time with a helpful message.
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Pass api_key=... to the Reader "
                "or export OPENAI_API_KEY in your environment."
            )
        from openai import OpenAI  # lazy: avoids forcing import at package load
        self._client = OpenAI(api_key=key, base_url=base_url)

    def warm(self) -> None:
        """No-op for the cloud backend (no weights to load)."""

    def synthesize(
        self,
        text: Union[str, Iterable[str]],
        *,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Iterator[AudioChunk]:
        v = _resolve_voice(voice) if voice is not None else self.voice
        instr = instructions if instructions is not None else self.instructions

        if isinstance(text, str):
            yield from self._synth_one(text, v, instr, sentence_index=0, is_last_sentence=True)
            return

        # Token-stream input: drain sentences as they complete, synthesize each.
        # Mirrors the local backend's behavior so the Reader API is consistent.
        buf, i = "", 0
        for piece in text:
            if not piece: continue
            buf += piece
            sents, buf = drain_sentences(buf)
            for s in sents:
                yield from self._synth_one(s, v, instr, sentence_index=i, is_last_sentence=False)
                i += 1
        tail = buf.strip()
        if tail:
            yield from self._synth_one(tail, v, instr, sentence_index=i, is_last_sentence=True)

    def _synth_one(
        self,
        sentence: str,
        voice: str,
        instructions: Optional[str],
        *,
        sentence_index: int,
        is_last_sentence: bool,
    ) -> Iterator[AudioChunk]:
        kwargs = {
            "model": self.model,
            "voice": voice,
            "input": sentence,
            "response_format": "pcm",
        }
        if instructions and self.model in _MODELS_WITH_INSTRUCTIONS:
            kwargs["instructions"] = instructions

        t0 = time.time()
        ttfa: Optional[float] = None
        chunk_index = 0

        with self._client.audio.speech.with_streaming_response.create(**kwargs) as response:
            for raw in response.iter_bytes(chunk_size=_STREAM_CHUNK_BYTES):
                if not raw:
                    continue
                # Guard against an odd-byte boundary at the chunk edge: trim
                # any trailing odd byte and stash it for the next iteration
                # would be ideal, but the OpenAI server sends whole frames so
                # in practice this is always even. We still defend.
                if len(raw) % BYTES_PER_SAMPLE:
                    raw = raw[: len(raw) - (len(raw) % BYTES_PER_SAMPLE)]
                int16 = np.frombuffer(raw, dtype="<i2")
                pcm = int16.astype(np.float32) / 32768.0
                if ttfa is None:
                    ttfa = time.time() - t0
                chunk_index += 1
                yield AudioChunk(
                    pcm=pcm,
                    sample_rate=SAMPLE_RATE,
                    is_final=False,
                    meta={
                        "backend": "openai",
                        "model": self.model,
                        "voice": voice,
                        "sentence_index": sentence_index,
                        "chunk_index": chunk_index,
                        "ttfa_sentence": ttfa,
                    },
                )

        if is_last_sentence:
            # Emit a zero-length terminal chunk so sinks that key off
            # is_final get a clean signal. Matches local backend semantics.
            yield AudioChunk(
                pcm=np.zeros(0, dtype=np.float32),
                sample_rate=SAMPLE_RATE,
                is_final=True,
                meta={
                    "backend": "openai",
                    "model": self.model,
                    "voice": voice,
                    "sentence_index": sentence_index,
                    "chunk_index": chunk_index + 1,
                    "ttfa_sentence": ttfa,
                },
            )
