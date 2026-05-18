"""Detector pipeline: pluggable inline content recognition.

A Detector scans text and returns a list of `Segment` (Claimed | Unclaimed).
The pipeline runs detectors in priority order; later detectors only see
prior passes' Unclaimed regions, so structural detectors (URLs, code, math)
naturally protect their content from substitution detectors (numbers, symbols).

To add a custom detector, implement the `Detector` protocol and add it to a
`Pipeline`. The most common pattern is to swap in a custom EquationDetector
with an LLM-backed reader; see `examples/groq_equation_reader.py`.
"""
from .types       import Segment, Claimed, Unclaimed, Detector
from .pipeline    import Pipeline, normalize_text
from .builtin     import (
    UrlDetector, InlineCodeDetector, SymbolDetector, NumberDetector,
    EquationDetector, DateDetector, PhoneDetector, AcronymDetector,
)
from .rule_reader import rule_based_reader

__all__ = [
    "Segment", "Claimed", "Unclaimed", "Detector",
    "Pipeline", "normalize_text",
    "UrlDetector", "InlineCodeDetector", "SymbolDetector",
    "NumberDetector", "EquationDetector", "DateDetector", "PhoneDetector",
    "AcronymDetector",
    "rule_based_reader",
    "default_pipeline",
]


def default_pipeline() -> Pipeline:
    """The standard set of detectors. Override per-Reader or per-call."""
    return Pipeline([
        EquationDetector(),       # priority 10
        UrlDetector(),            # priority 20
        InlineCodeDetector(),     # priority 30
        DateDetector(),           # priority 40
        PhoneDetector(),          # priority 50
        NumberDetector(),         # priority 60
        SymbolDetector(),         # priority 70
    ])
