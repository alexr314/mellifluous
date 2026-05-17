"""Pipeline orchestrator."""
from __future__ import annotations
from typing import Iterable

from .types import Segment, Claimed, Unclaimed, Detector


class Pipeline:
    """Runs a sequence of detectors over text.

    Sorted by `priority` (ascending). Each detector only sees `Unclaimed`
    regions from the prior pass, so earlier detectors protect their content.
    """
    def __init__(self, detectors: Iterable[Detector]):
        self.detectors = sorted(detectors, key=lambda d: d.priority)

    def process(self, text: str) -> list[Segment]:
        segments: list[Segment] = [Unclaimed(text)]
        for det in self.detectors:
            segments = self._apply(det, segments)
        return segments

    def _apply(self, detector: Detector, segments: list[Segment]) -> list[Segment]:
        new: list[Segment] = []
        for seg in segments:
            if isinstance(seg, Claimed):
                new.append(seg)
                continue
            sub = detector.scan(seg.raw)
            if not sub:
                new.append(seg)
                continue
            if "".join(s.raw for s in sub) != seg.raw:
                raise RuntimeError(
                    f"detector {detector.name!r} did not preserve text "
                    f"(in={seg.raw!r}, joined_out={''.join(s.raw for s in sub)!r})"
                )
            new.extend(sub)
        return new


def normalize_text(pipeline: Pipeline, text: str) -> str:
    """Run pipeline and concatenate to a final spoken string."""
    return "".join(
        s.spoken if isinstance(s, Claimed) else s.raw
        for s in pipeline.process(text)
    )
