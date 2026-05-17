"""Bridge: Utterance iterator -> AudioChunk iterator with collapsing pauses."""
from __future__ import annotations
from typing import Iterable, Iterator, Optional

import numpy as np

from .base   import Backend
from .types  import AudioChunk
from ..utterance import Utterance


def _silence(ms: int, sample_rate: int) -> Optional[AudioChunk]:
    if ms <= 0:
        return None
    n = max(1, int(round(ms / 1000 * sample_rate)))
    return AudioChunk(
        pcm=np.zeros(n, dtype=np.float32),
        sample_rate=sample_rate,
        is_final=False,
        meta={"silence_ms": ms},
    )


def synthesize_utterances(
    backend: Backend,
    utterances: Iterable[Utterance],
    *,
    sample_rate: Optional[int] = None,
) -> Iterator[AudioChunk]:
    """Yield AudioChunks for a stream of Utterances.

    Between two utterances we insert silence of `max(prev.post, this.pre)`
    so per-span pause requests don't double-count.
    """
    sr = sample_rate if sample_rate is not None else backend.sample_rate
    pending_pause = 0
    for u in utterances:
        gap = max(pending_pause, u.pre_pause_ms)
        s = _silence(gap, sr)
        if s is not None:
            yield s
        pending_pause = 0

        if u.text:
            for chunk in backend.synthesize(u.text):
                sr = chunk.sample_rate
                yield chunk

        pending_pause = u.post_pause_ms

    s = _silence(pending_pause, sr)
    if s is not None:
        yield s
