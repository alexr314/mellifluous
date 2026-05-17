"""AudioChunk: the unit of streaming audio."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """One piece of streaming audio.

    `pcm` is float32 mono at `sample_rate` Hz, the format the model emits.
    Sinks (play/write_wav/HTTP) accept iterators of these.
    """
    pcm: np.ndarray
    sample_rate: int = 24000
    is_final: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate
