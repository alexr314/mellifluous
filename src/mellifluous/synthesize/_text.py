"""Shared text utilities: sentence splitter for streaming token inputs.

Used by all backends so a token stream (e.g. an LLM that yields one token at
a time) gets carved into sentences before being sent to the TTS engine. Each
backend has its own per-sentence behavior, but the carving rules are common.
"""
from __future__ import annotations
import re


_ABBRS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St",
          "vs", "etc", "e.g", "i.e", "U.S", "U.K")
_MASK_ABBR    = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBRS) + r")\.")
_MASK_DECIMAL = re.compile(r"(\d)\.(?=\d)")
_DOT_SENTINEL = "\x00"
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")


def drain_sentences(buf: str) -> tuple[list[str], str]:
    """Pull complete sentences out of `buf`, return (sentences, remainder).

    Periods inside abbreviations ("Dr.") and decimals ("3.14") are masked so
    they don't trigger a split. The remainder is whatever didn't end in a
    terminator yet — feed more text and call again.
    """
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
