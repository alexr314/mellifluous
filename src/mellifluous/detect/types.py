"""Core types for the detector pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Claimed:
    """A region of text a detector handled. `spoken` replaces `raw` in output."""
    raw: str
    spoken: str
    detector: str


@dataclass(frozen=True)
class Unclaimed:
    """A region no detector has handled (or chose to leave alone)."""
    raw: str


Segment = Claimed | Unclaimed


@runtime_checkable
class Detector(Protocol):
    """Walks text left-to-right and emits Segments.

    Implementations must preserve text: ''.join(s.raw for s in scan(text)) == text.
    Lower `priority` runs first. Conventional ranges:
      10..30  structural / containing (math, links, code)
      50..70  substitutions inside prose (numbers, symbols)
      90..100 cleanup
    """
    name: str
    priority: int
    def scan(self, text: str) -> list[Segment]: ...
