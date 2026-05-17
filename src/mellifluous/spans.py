"""Spans: the intermediate representation produced by the parser.

A document parses into a flat list of Spans. The parser only labels what each
piece *is*; pause/pronunciation decisions happen in the vocalizer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Span:
    text: str

@dataclass(frozen=True)
class Paragraph(Span): pass

@dataclass(frozen=True)
class Heading(Span):
    level: int                       # 1..6

@dataclass(frozen=True)
class ListItem(Span):
    ordered: bool
    index: int                       # 1-based position within parent list
    is_first: bool = False
    is_last: bool = False

@dataclass(frozen=True)
class BlockQuote(Span): pass

@dataclass(frozen=True)
class CodeBlock(Span):
    lang: Optional[str] = None
    line_count: int = 0

@dataclass(frozen=True)
class Equation(Span):
    """Display math ($$...$$ or \\[...\\] at block level)."""

@dataclass(frozen=True)
class HorizontalRule(Span): pass

@dataclass(frozen=True)
class TableSummary(Span):
    n_rows: int = 0
    n_cols: int = 0
    headers: tuple[str, ...] = ()      # column headers (always present in md tables)
    rows:    tuple[tuple[str, ...], ...] = ()  # body rows (data only, not header)


Document = list[Span]
