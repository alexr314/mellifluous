"""Streamer: model wrapper that yields AudioChunks."""
from __future__ import annotations
import logging
import re
import time
from typing import Iterable, Iterator, Optional, Union

import numpy as np

from .types  import AudioChunk
from .voices import Voice, Clone, Preset
from .models import MODELS, DEFAULT_MODEL, ModelSpec

log = logging.getLogger(__name__)


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


# --- sentence splitter for incremental text streams (LLM token streams) ---

_ABBRS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
          "vs", "etc", "e.g", "i.e", "U.S", "U.K")
_MASK_ABBR    = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBRS) + r")\.")
_MASK_DECIMAL = re.compile(r"(\d)\.(?=\d)")
_DOT_SENTINEL = "\x00"
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")


# Heuristic: when a sentence ends in a very short word ("x.", "y.", "a.",
# "of a.") the TTS model tends to clip the last word. Replacing the trailing
# period with "......" gives the model more landing room. The cutoff of 3
# characters is empirical, picked to catch single-letter variables and
# short closing words ("of a." etc.) without affecting longer sentences.
_SHORT_FINAL_WORD = re.compile(r"(\b\w{1,3})([.!?])\s*$")

def _pad_short_final_word(text: str) -> str:
    return _SHORT_FINAL_WORD.sub(lambda m: m.group(1) + "......", text)


def _drain_sentences(buf: str) -> tuple[list[str], str]:
    masked = _MASK_ABBR.sub(lambda m: m.group(1) + _DOT_SENTINEL, buf)
    masked = _MASK_DECIMAL.sub(lambda m: m.group(1) + _DOT_SENTINEL, masked)
    out, last = [], 0
    for m in _SENTENCE_END.finditer(masked):
        end = m.end()
        s = buf[last:end].strip()
        if s:
            out.append(s)
        last = end
    return out, buf[last:]


class Streamer:
    """Wraps a Qwen3-TTS model variant and yields AudioChunks for input text.

    The same Streamer object is reused across calls; the model loads once.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: Optional[Voice] = None,
        *,
        streaming_interval: float = 0.5,
        temperature: float = 0.9,
    ):
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}. choices: {list(MODELS)}")
        self.spec = MODELS[model]
        self.voice = voice
        self.streaming_interval = streaming_interval
        self.temperature = temperature
        if voice is not None:
            self._check_voice(voice)

    def _check_voice(self, voice: Voice) -> None:
        if self.spec.kind == "clone" and not isinstance(voice, Clone):
            raise TypeError(f"{self.spec.id} is a cloning model; pass a Clone voice")
        if self.spec.kind == "preset" and not isinstance(voice, Preset):
            raise TypeError(f"{self.spec.id} is a preset model; pass a Preset voice")

    # ---------- public ----------

    def warm(self) -> None:
        """Load the model weights into memory so the next synthesize() is fast."""
        _load(self.spec)

    def synthesize(
        self,
        text: Union[str, Iterable[str]],
        *,
        voice: Optional[Voice] = None,
    ) -> Iterator[AudioChunk]:
        """Synthesize `text` and yield AudioChunks as they're generated.

        `text` can be a string or an iterable of strings (e.g. an LLM token
        stream). With an iterable, the streamer buffers tokens until a sentence
        terminator appears, then synthesizes one sentence at a time.
        """
        v = voice or self.voice
        if v is None:
            raise ValueError("no voice set; pass voice= here or in __init__")
        self._check_voice(v)

        if isinstance(text, str):
            yield from self._synth_one(text, v, sentence_index=0, is_last_sentence=True)
            return

        buf, i = "", 0
        for piece in text:
            if not piece: continue
            buf += piece
            sents, buf = _drain_sentences(buf)
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
                    "model": self.spec.id,
                    "sentence_index": sentence_index,
                    "chunk_index": chunk_index,
                    "ttfa_sentence": ttfa,
                },
            )
