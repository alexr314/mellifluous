"""Backend abstraction: every TTS engine (MLX local, OpenAI cloud, ...)
implements the same `Backend` interface so the rest of mellifluous doesn't
care which one is producing audio.

A backend takes text (string or token iterator) and yields `AudioChunk`s. It
owns its own voice resolution, model loading, and any engine-specific knobs.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Optional, Union

from .types  import AudioChunk
from .voices import Voice


class Backend(ABC):
    """The contract every TTS backend satisfies.

    Subclasses set `sample_rate` on the instance to match what they emit.
    """
    sample_rate: int = 24000

    @abstractmethod
    def warm(self) -> None:
        """Pay any one-time setup cost (model load, network handshake) now."""

    @abstractmethod
    def synthesize(
        self,
        text: Union[str, Iterable[str]],
        *,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Iterator[AudioChunk]:
        """Synthesize `text` and yield AudioChunks as they're generated.

        `text` may be a string or an iterable of strings (e.g. an LLM token
        stream); iterables are split into sentences and synthesized one
        sentence at a time. `voice` and `instructions` override per-call any
        defaults set in the constructor. Backends that don't support
        `instructions` ignore it.
        """


def make_backend(engine: str, model: str, **kwargs: Any) -> Backend:
    """Construct a backend by engine name.

    Supported engines:
      - "openai" — OpenAI cloud TTS (gpt-4o-mini-tts, tts-1, tts-1-hd)
      - "local"  — Qwen3-TTS on Apple Silicon via mlx-audio (macOS only)
    """
    if engine == "openai":
        from .backends.openai import OpenAIBackend
        return OpenAIBackend(model=model, **kwargs)
    if engine == "local":
        from .backends.local_mlx import LocalMLXBackend
        return LocalMLXBackend(model=model, **kwargs)
    raise ValueError(
        f"unknown engine {engine!r}. choices: 'openai', 'local'"
    )
