"""Local MLX backend: Qwen3-TTS via mlx-audio (macOS Apple Silicon only).

This is the original mellifluous TTS path, repackaged behind the `Backend`
interface. Same model variants, same voice handling (Clone for *Base
variants, Preset for CustomVoice variants), same streaming-interval knob.

The `mlx_audio` import is intentionally deferred until `warm()` or
`synthesize()` actually runs, so the module imports fine on non-darwin
systems (the rest of mellifluous can still be used with another backend).
"""
from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass
from pathlib  import Path
from typing   import Iterable, Iterator, Optional, Union

import numpy as np

from ..base   import Backend
from ..types  import AudioChunk
from ..voices import Voice, Clone, Preset
from .._text  import drain_sentences

log = logging.getLogger(__name__)


# --- model registry -------------------------------------------------------

ModelKind = str  # "clone" | "preset"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str            # HuggingFace repo (mlx-audio compatible)
    kind: ModelKind


MODELS: dict[str, ModelSpec] = {
    m.id: m for m in [
        ModelSpec("qwen-1.7b-8bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",        "clone"),
        ModelSpec("qwen-1.7b-6bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit",        "clone"),
        ModelSpec("qwen-1.7b-4bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",        "clone"),
        ModelSpec("qwen-0.6b-8bit",        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",        "clone"),
        ModelSpec("qwen-1.7b-custom-8bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "preset"),
    ]
}

DEFAULT_MODEL = "qwen-1.7b-8bit"

PRESET_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]


# --- model cache ----------------------------------------------------------

# Process-wide model cache. We keep at most one loaded at a time, because
# Apple Silicon unified memory is shared with the rest of your apps.
_loaded_key: Optional[str] = None
_loaded_model = None


def _load(spec: ModelSpec):
    global _loaded_key, _loaded_model
    if _loaded_key == spec.id:
        return _loaded_model
    from mlx_audio.tts.utils import load_model  # lazy: heavy import
    log.info("loading %s (%s)", spec.id, spec.repo)
    t = time.time()
    _loaded_model = load_model(spec.repo)
    _loaded_key   = spec.id
    log.info("  loaded in %.1fs", time.time() - t)
    return _loaded_model


# --- short-final-word pad -------------------------------------------------

# Heuristic: when a sentence ends in a very short word ("x.", "y.", "a.",
# "of a.") the TTS model tends to clip the last word. Replacing the trailing
# period with "......" gives the model more landing room. The cutoff of 3
# characters is empirical, picked to catch single-letter variables and
# short closing words ("of a." etc.) without affecting longer sentences.
_SHORT_FINAL_WORD = re.compile(r"(\b\w{1,3})([.!?])\s*$")


def _pad_short_final_word(text: str) -> str:
    return _SHORT_FINAL_WORD.sub(lambda m: m.group(1) + "......", text)


# --- voice resolution -----------------------------------------------------

def _default_voices_dir() -> Path:
    import os
    env = os.environ.get("MELLIFLUOUS_VOICES_DIR")
    if env:
        return Path(env).expanduser()
    # repo_root/voices  — this file is at
    #   repo_root/src/mellifluous/synthesize/backends/local_mlx.py
    return Path(__file__).resolve().parents[4] / "voices"


def list_voices(voices_dir: Optional[Path] = None) -> list[str]:
    d = voices_dir or _default_voices_dir()
    if not d.exists():
        return []
    return sorted(
        sub.name for sub in d.iterdir()
        if sub.is_dir() and (sub / "sample.wav").exists()
    )


def find_voice(name: str, voices_dir: Optional[Path] = None) -> Path:
    d = voices_dir or _default_voices_dir()
    sample = d / name / "sample.wav"
    if not sample.exists():
        available = list_voices(d)
        raise FileNotFoundError(
            f"voice {name!r} not found at {sample}. "
            f"available: {available or '(none. Add a sample.wav under voices/<name>/)'}"
        )
    return sample


# --- backend --------------------------------------------------------------

class LocalMLXBackend(Backend):
    """Qwen3-TTS via mlx-audio. macOS Apple Silicon only."""

    sample_rate = 24000

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: Optional[Union[str, Voice]] = None,
        voices_dir: Optional[Path] = None,
        streaming_interval: float = 0.5,
        temperature: float = 0.9,
        # `instructions` accepted for API uniformity with OpenAIBackend; ignored.
        instructions: Optional[str] = None,
    ):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}. choices: {sorted(MODELS)}")
        self.spec = MODELS[model]
        self.voices_dir = voices_dir or _default_voices_dir()
        self.voice = self._resolve_voice(voice)
        self.streaming_interval = streaming_interval
        self.temperature = temperature

    def _resolve_voice(self, voice: Optional[Union[str, Voice]]) -> Voice:
        if isinstance(voice, (Clone, Preset)):
            self._check_voice(voice)
            return voice
        if self.spec.kind == "clone":
            if voice is None:
                available = list_voices(self.voices_dir)
                if not available:
                    raise FileNotFoundError(
                        f"no voices found in {self.voices_dir}. "
                        "Add one at voices/<name>/sample.wav, or pass voice=Clone(...)."
                    )
                name = available[0]
            else:
                name = voice
            return Clone(ref_audio=str(find_voice(name, self.voices_dir)))
        # preset model
        raise TypeError(
            f"model {self.spec.id!r} is a preset model. Pass voice=Preset('name'). "
            f"Available presets: {PRESET_SPEAKERS}"
        )

    def _check_voice(self, voice: Voice) -> None:
        if self.spec.kind == "clone" and not isinstance(voice, Clone):
            raise TypeError(f"{self.spec.id} is a cloning model; pass a Clone voice")
        if self.spec.kind == "preset" and not isinstance(voice, Preset):
            raise TypeError(f"{self.spec.id} is a preset model; pass a Preset voice")

    # ---------- public ----------

    def warm(self) -> None:
        _load(self.spec)

    def synthesize(
        self,
        text: Union[str, Iterable[str]],
        *,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,  # ignored
    ) -> Iterator[AudioChunk]:
        v = self._resolve_voice(voice) if voice is not None else self.voice

        if isinstance(text, str):
            yield from self._synth_one(text, v, sentence_index=0, is_last_sentence=True)
            return

        buf, i = "", 0
        for piece in text:
            if not piece: continue
            buf += piece
            sents, buf = drain_sentences(buf)
            for s in sents:
                yield from self._synth_one(s, v, sentence_index=i, is_last_sentence=False)
                i += 1
        tail = buf.strip()
        if tail:
            yield from self._synth_one(tail, v, sentence_index=i, is_last_sentence=True)

    # ---------- internal ----------

    def _synth_one(self, sentence: str, voice: Voice, *,
                   sentence_index: int, is_last_sentence: bool) -> Iterator[AudioChunk]:
        model = _load(self.spec)
        # The Qwen3-TTS model uses sentence-final punctuation as a cue for
        # prosodic closure. Short utterances without a terminator (e.g. a
        # markdown heading like "Lists") get clipped: the model decides it's
        # mid-thought and trails off. Appending a period if there's none
        # gives it a complete-sounding ending.
        if sentence and sentence[-1] not in '.!?"\')]':
            sentence = sentence + "."
        # Even with a period, sentences that end in a single short word
        # (e.g. "x arrow y." or "f of b minus f of a.") get the same clip.
        # Swap the terminating period for a doubled ellipsis when the final
        # word is short. Empirically the model gives those words more room.
        sentence = _pad_short_final_word(sentence)
        t0 = time.time()
        ttfa = None
        chunk_index = 0

        if isinstance(voice, Clone):
            stream = model.generate(
                text=sentence,
                ref_audio=voice.ref_audio,
                ref_text=voice.ref_text,
                temperature=self.temperature,
                stream=True,
                streaming_interval=self.streaming_interval,
            )
        else:
            kwargs = dict(
                text=sentence,
                speaker=voice.speaker,
                language=voice.language,
                temperature=self.temperature,
                stream=True,
                streaming_interval=self.streaming_interval,
            )
            if voice.instruct:
                kwargs["instruct"] = voice.instruct
            stream = model.generate_custom_voice(**kwargs)

        for r in stream:
            pcm = np.array(r.audio, dtype=np.float32, copy=False)
            if ttfa is None:
                ttfa = time.time() - t0
            chunk_index += 1
            yield AudioChunk(
                pcm=pcm,
                sample_rate=int(getattr(r, "sample_rate", 24000)),
                is_final=bool(getattr(r, "is_final_chunk", False)) and is_last_sentence,
                meta={
                    "backend": "local",
                    "model": self.spec.id,
                    "sentence_index": sentence_index,
                    "chunk_index": chunk_index,
                    "ttfa_sentence": ttfa,
                },
            )
