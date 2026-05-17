"""Utterance: the vocalizer's output. Speakable text with requested pauses."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Utterance:
    """One unit the TTS will speak, with pauses requested around it.

    Adjacent utterances collapse overlap: the silence between A and B is
    `max(A.post_pause_ms, B.pre_pause_ms)`. This lets authors set per-span
    pauses independently without double-counting.
    """
    text: str
    pre_pause_ms:  int = 0
    post_pause_ms: int = 0
    role: str = ""        # debug/log label like "heading.1" or "paragraph"
