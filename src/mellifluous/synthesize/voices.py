"""Voice selection types."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Clone:
    """A voice cloned from a reference audio clip (Base model variants)."""
    ref_audio: str
    ref_text: Optional[str] = None     # transcript; whisper-auto if None


@dataclass(frozen=True)
class Preset:
    """A built-in named speaker (CustomVoice model variants)."""
    speaker: str                       # one of PRESET_SPEAKERS
    instruct: Optional[str] = None     # emotion/style cue
    language: str = "English"


Voice = Clone | Preset
