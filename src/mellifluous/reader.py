"""Reader: the high-level entry point.

    from mellifluous import Reader
    r = Reader()                                # defaults to OpenAI gpt-4o-mini-tts, voice "ash"
    r.speak("# Hello\\n\\nThis is a *test*.")

    # Use the local Qwen3-TTS backend (macOS Apple Silicon only):
    r = Reader(engine="local", model="qwen-1.7b-8bit", voice="alex")
    r.warm()
    r.speak("...")

    # Override per-call:
    r.speak("Heads up — server alert!", instructions="urgent, attention-grabbing")
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing  import Any, Iterable, Iterator, Optional, Union

from .parse      import parse_markdown
from .vocalize   import vocalize, Policy
from .utterance  import Utterance
from .synthesize import (
    AudioChunk, Backend, Voice, make_backend,
    synthesize_utterances, play, write_wav,
)


# Engine-specific defaults for `model`. When the caller doesn't pass model=
# explicitly, we look up by engine here.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini-tts",
    "local":  "qwen-1.7b-8bit",
}


def _default_engine() -> str:
    """Pick a backend that will actually work in the current environment.

    Preference order:
      1. macOS Apple Silicon with mlx_audio installed -> "local" (offline,
         voice cloning, sounds better to the author's ear)
      2. otherwise -> "openai" (works anywhere, needs OPENAI_API_KEY; if
         the key is missing the OpenAIBackend raises a clear error)
    """
    if sys.platform == "darwin":
        try:
            import mlx_audio  # noqa: F401
            return "local"
        except ImportError:
            pass
    return "openai"


# --- voice directory helpers (local-MLX-specific, re-exported for convenience) ---

def list_voices(voices_dir: Optional[Path] = None) -> list[str]:
    """Return the names of voices found in the local voices/ directory.

    Voices are subdirectories containing `sample.wav`. Only meaningful for
    the `local` engine; OpenAI voices are presets, not files.
    """
    from .synthesize.backends.local_mlx import list_voices as _ls
    return _ls(voices_dir)


def find_voice(name: str, voices_dir: Optional[Path] = None) -> Path:
    """Return the path to `voices/<name>/sample.wav`. Raises if missing.

    Only meaningful for the `local` engine.
    """
    from .synthesize.backends.local_mlx import find_voice as _fv
    return _fv(name, voices_dir)


# --- Reader -----------------------------------------------------------------

class Reader:
    """High-level markdown -> speech.

    Args:
        engine:   "local" or "openai". If omitted, picks "local" on macOS
                  Apple Silicon when mlx-audio is installed; otherwise
                  "openai" (which then needs OPENAI_API_KEY).
        model:    backend-specific model id. Defaults: gpt-4o-mini-tts (openai)
                  or qwen-1.7b-8bit (local).
        voice:    OpenAI: a preset name like "ash", "nova", "sage", etc.
                  Local: a voice name from voices/<name>/, or a Voice instance.
                  None picks a sensible default per backend.
        instructions: OpenAI gpt-4o-mini-tts only — tone steering string,
                  e.g. "calm, conversational assistant". Set a Reader-level
                  default; override per-call via speak()/synthesize().
        policy:   Vocalization policy (pauses, verbosity, detectors).
        voices_dir: Local engine only — override the voices/ directory
                  (or set MELLIFLUOUS_VOICES_DIR).
        api_key:  OpenAI only — defaults to OPENAI_API_KEY env var.
        base_url: OpenAI only — for proxies / Azure / compat APIs.

    Backend-specific kwargs (instructions, api_key, base_url, voices_dir,
    streaming_interval, temperature) are forwarded to the chosen backend.
    """

    def __init__(
        self,
        *,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
        policy: Optional[Policy] = None,
        **backend_kwargs: Any,
    ):
        self.engine = engine if engine is not None else _default_engine()
        self.model = model if model is not None else _DEFAULT_MODELS.get(self.engine)
        if self.model is None:
            raise ValueError(
                f"no default model registered for engine {engine!r}; pass model=..."
            )
        self.policy = policy or Policy()
        self.backend: Backend = make_backend(
            engine=self.engine,
            model=self.model,
            voice=voice,
            instructions=instructions,
            **backend_kwargs,
        )
        # Expose the resolved voice on the Reader for tests / introspection.
        # Each backend stashes its own (a str for openai, a Voice for local).
        self.voice = getattr(self.backend, "voice", voice)

    # ---------- public ----------

    def warm(self) -> None:
        """Pay one-time setup costs now (model load for local; no-op for openai)."""
        self.backend.warm()

    def utterances(self, markdown_text: str) -> Iterator[Utterance]:
        """Parse markdown and yield Utterances. No audio."""
        return vocalize(parse_markdown(markdown_text), policy=self.policy)

    def synthesize(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Iterator[AudioChunk]:
        """Yield AudioChunks for the given text.

        If `as_markdown` is True (default), `text_or_iter` is treated as a
        markdown string and pause-aware Utterances are produced. If False,
        the text is sent directly to TTS (useful for already-clean strings
        or LLM token streams that don't carry markdown structure).

        `voice` and `instructions` override Reader defaults for this call.
        """
        if as_markdown and isinstance(text_or_iter, str):
            # Markdown path: utterances flow through the bridge, which
            # doesn't know about voice/instructions overrides. Apply them
            # by swapping the backend's defaults for the duration of the
            # call. Cheap and avoids plumbing through the bridge.
            return self._synthesize_markdown(
                text_or_iter, voice=voice, instructions=instructions
            )
        return self.backend.synthesize(
            text_or_iter, voice=voice, instructions=instructions
        )

    def speak(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> None:
        """Synthesize and play on the default audio output. Blocks until done."""
        play(self.synthesize(
            text_or_iter, as_markdown=as_markdown,
            voice=voice, instructions=instructions,
        ))

    def to_wav(
        self,
        text_or_iter: Union[str, Iterable[str]],
        path: str | Path,
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Path:
        """Synthesize and write a WAV file. Returns the path."""
        return write_wav(path, self.synthesize(
            text_or_iter, as_markdown=as_markdown,
            voice=voice, instructions=instructions,
        ))

    # ---------- internal ----------

    def _synthesize_markdown(
        self,
        markdown_text: str,
        *,
        voice: Optional[Union[str, Voice]],
        instructions: Optional[str],
    ) -> Iterator[AudioChunk]:
        # The bridge calls backend.synthesize(utt.text) without per-call
        # overrides. To honor per-call voice/instructions on the markdown
        # path, swap the backend's defaults for the duration of this call.
        backend = self.backend
        saved_voice = getattr(backend, "voice", None)
        saved_instr = getattr(backend, "instructions", None)
        if voice is not None and hasattr(backend, "voice"):
            backend.voice = backend._resolve_voice(voice) if hasattr(backend, "_resolve_voice") else voice
        if instructions is not None and hasattr(backend, "instructions"):
            backend.instructions = instructions
        try:
            yield from synthesize_utterances(backend, self.utterances(markdown_text))
        finally:
            if hasattr(backend, "voice"):
                backend.voice = saved_voice
            if hasattr(backend, "instructions"):
                backend.instructions = saved_instr
