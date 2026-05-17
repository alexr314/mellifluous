"""Reader: the high-level entry point.

    from mellifluous import Reader
    r = Reader()                          # uses the first voice in voices/
    r.warm()                              # optional: load model now, not on first speak
    r.speak("# Hello\\n\\nThis is a *test*.")

    # or get the chunks yourself
    for chunk in r.synthesize("Some text"):
        ...

    # or just turn markdown into utterances without any audio
    for utt in r.utterances("# Title\\n\\nA paragraph."):
        print(utt)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from .parse      import parse_markdown
from .vocalize   import vocalize, Policy
from .utterance  import Utterance
from .synthesize import (
    AudioChunk, Streamer, Clone, Preset, Voice,
    DEFAULT_MODEL, MODELS, PRESET_SPEAKERS,
    synthesize_utterances, play, write_wav,
)


# Default voice library is the `voices/` directory at the repo root. Override
# with the MELLIFLUOUS_VOICES_DIR environment variable.
def _default_voices_dir() -> Path:
    env = os.environ.get("MELLIFLUOUS_VOICES_DIR")
    if env:
        return Path(env).expanduser()
    # repo_root/voices  (this file is at  repo_root/src/mellifluous/reader.py)
    return Path(__file__).resolve().parents[2] / "voices"


def list_voices(voices_dir: Optional[Path] = None) -> list[str]:
    """Return the names of all voices found in the voices/ directory.

    A voice is any subdirectory containing `sample.wav`.
    Sorted alphabetically.
    """
    d = voices_dir or _default_voices_dir()
    if not d.exists():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "sample.wav").exists()
    )


def find_voice(name: str, voices_dir: Optional[Path] = None) -> Path:
    """Return the path to `voices/<name>/sample.wav`. Raises if missing."""
    d = voices_dir or _default_voices_dir()
    sample = d / name / "sample.wav"
    if not sample.exists():
        available = list_voices(d)
        raise FileNotFoundError(
            f"voice {name!r} not found at {sample}. "
            f"available: {available or '(none. Add a sample.wav under voices/<name>/)'}"
        )
    return sample


class Reader:
    """High-level markdown -> speech.

    Args:
        voice: voice name from the voices/ directory, or a Voice instance,
               or None to pick the first available voice.
        model: model id from synthesize.MODELS (default: '1.7b-8bit', cloning).
        policy: vocalization policy (pauses, verbosity, detectors).
        voices_dir: override the voices directory (or set MELLIFLUOUS_VOICES_DIR).
    """

    def __init__(
        self,
        voice: Optional[Union[str, Voice]] = None,
        *,
        model: str = DEFAULT_MODEL,
        policy: Optional[Policy] = None,
        voices_dir: Optional[Path] = None,
    ):
        self.voices_dir = voices_dir or _default_voices_dir()
        self.policy = policy or Policy()
        self.voice = self._resolve_voice(voice, model)
        self.streamer = Streamer(model=model, voice=self.voice)

    # ---------- voice resolution ----------

    def _resolve_voice(self, voice, model: str) -> Voice:
        if isinstance(voice, (Clone, Preset)):
            return voice
        kind = MODELS[model].kind
        if voice is None:
            available = list_voices(self.voices_dir)
            if not available:
                raise FileNotFoundError(
                    f"no voices found in {self.voices_dir}. "
                    "Add one at voices/<name>/sample.wav, or pass voice=Preset(...)."
                )
            name = available[0]
        else:
            name = voice
        if kind == "clone":
            return Clone(ref_audio=str(find_voice(name, self.voices_dir)))
        raise TypeError(
            f"model {model!r} is a preset model. Pass voice=Preset('name'). "
            f"Available presets: {PRESET_SPEAKERS}"
        )

    # ---------- public ----------

    def warm(self) -> None:
        """Load the model weights now. Optional but recommended for low-latency apps."""
        self.streamer.warm()

    def utterances(self, markdown_text: str) -> Iterator[Utterance]:
        """Parse markdown and yield Utterances. No audio."""
        return vocalize(parse_markdown(markdown_text), policy=self.policy)

    def synthesize(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
    ) -> Iterator[AudioChunk]:
        """Yield AudioChunks for the given text.

        If `as_markdown` is True (default), `text_or_iter` is treated as a
        markdown string and pause-aware Utterances are produced. If False,
        the text is sent directly to TTS (useful for already-clean strings
        or LLM token streams that don't carry markdown structure).
        """
        if as_markdown and isinstance(text_or_iter, str):
            return synthesize_utterances(self.streamer, self.utterances(text_or_iter))
        return self.streamer.synthesize(text_or_iter)

    def speak(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
    ) -> None:
        """Synthesize and play on the default audio output. Blocks until done."""
        play(self.synthesize(text_or_iter, as_markdown=as_markdown))

    def to_wav(
        self,
        text_or_iter: Union[str, Iterable[str]],
        path: str | Path,
        *,
        as_markdown: bool = True,
    ) -> Path:
        """Synthesize and write a WAV file. Returns the path."""
        return write_wav(path, self.synthesize(text_or_iter, as_markdown=as_markdown))
