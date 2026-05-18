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


# --- markdown block splitter (for streaming markdown input) --------------

_FENCE_LINE = re.compile(r"^\s{0,3}(`{3,}|~{3,})", re.MULTILINE)


def drain_markdown_blocks(buf: str) -> tuple[list[str], str]:
    """Pull complete markdown blocks out of `buf`, return (blocks, remainder).

    A block is everything up to (and including) the next blank line that is
    NOT inside a fenced code block. The remainder is whatever is left after
    the last completed block.

    Used by the streaming markdown path: as LLM tokens arrive, accumulate
    them in a buffer, drain whatever blocks have completed, parse and speak
    each block while the next one is still being generated.

    Fenced code block handling: a line that starts (with 0-3 leading
    spaces) with ``` or ~~~ toggles 'inside code block' state. Blank lines
    inside a code block do not split.

    Tables, lists, and blockquotes are handled implicitly: they only end at
    a blank line in CommonMark, which is exactly the boundary we cut on.
    """
    blocks: list[str] = []
    pos = 0
    inside_fence = False
    block_start = 0
    n = len(buf)

    # Walk line by line so we can track fence state.
    while pos < n:
        nl = buf.find("\n", pos)
        line_end = n if nl == -1 else nl + 1
        line = buf[pos:line_end]

        if _FENCE_LINE.match(line):
            inside_fence = not inside_fence

        # A blank line outside a fence terminates the current block.
        if not inside_fence and line.strip() == "" and pos > block_start:
            block = buf[block_start:pos]
            if block.strip():
                blocks.append(block)
            block_start = line_end

        if nl == -1:
            # Incomplete trailing line; don't emit it yet (might be middle
            # of a sentence). Wait for more input.
            break
        pos = nl + 1

    remainder = buf[block_start:]
    return blocks, remainder


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
