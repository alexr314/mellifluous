"""Vocalizer: Document (list of Spans) -> Iterator[Utterance].

Each Utterance carries `pre_pause_ms` and `post_pause_ms`. The downstream
synthesizer is responsible for the collapsing rule between adjacent pauses:
silence between A and B = max(A.post_pause_ms, B.pre_pause_ms).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .spans import (
    Paragraph, Heading, ListItem, BlockQuote,
    CodeBlock, Equation, HorizontalRule, TableSummary, Document,
)
from .utterance import Utterance
from .detect import Pipeline, default_pipeline, normalize_text


@dataclass
class Policy:
    """Pause and rendering rules. Override fields to taste."""
    # Pauses (milliseconds).
    heading_pre: dict[int, int] = field(default_factory=lambda: {
        1: 1200, 2: 1000, 3: 800, 4: 600, 5: 500, 6: 400,
    })
    heading_post: dict[int, int] = field(default_factory=lambda: {
        1: 500, 2: 450, 3: 400, 4: 350, 5: 300, 6: 250,
    })
    paragraph_post:  int = 600
    list_item_post:  int = 250
    list_post:       int = 500       # extra silence after the last item
    blockquote_pre:  int = 500
    blockquote_post: int = 500
    code_pre:        int = 400
    code_post:       int = 400
    equation_pre:    int = 400
    equation_post:   int = 400
    rule_pre:        int = 700
    rule_post:       int = 700
    table_pre:       int = 400
    table_post:      int = 400

    # Verbosity knobs.
    say_heading_level: bool = False
    code_summary:      bool = True
    skip_code:         bool = True
    # Horizontal rules ('---') default to a silent pause -- announcing them
    # as "Section break" sounded jarring in practice. Set True to keep the
    # default; False brings back the spoken announcement.
    skip_horizontal_rule_sound: bool = True
    # If a table has at most this many body rows, read each row aloud paired
    # with column headers. Larger tables get a header-only summary.
    table_max_rows_to_read: int = 0

    # Detector pipeline applied to all spoken text. Swap for custom handlers.
    detectors: Pipeline = field(default_factory=default_pipeline)


DEFAULT_POLICY = Policy()


def vocalize(doc: Document, policy: Optional[Policy] = None) -> Iterator[Utterance]:
    p = policy or DEFAULT_POLICY

    def _norm(text: str) -> str:
        return normalize_text(p.detectors, text) if text else text

    for span in doc:
        if isinstance(span, Heading):
            text = _norm(span.text)
            if not text: continue
            if p.say_heading_level:
                prefix = {1: "Section.", 2: "Subsection.", 3: "Sub-subsection."}.get(span.level, "")
                if prefix:
                    text = f"{prefix} {text}"
            # Headings often lack terminal punctuation and the TTS model
            # tends to clip the last syllable as a result. A doubled ellipsis
            # (six dots) gives the model a "trailing off" cue that lands
            # more cleanly than a hard period; the long post-pause then
            # covers any extra wind-down. Empirically chosen from A/B tests
            # on several short headings; the BPE tokenizer treats '......'
            # as a distinct token from '...' so this isn't just "more dots".
            if text and text[-1] not in '.!?"\')]':
                text = text + "......"
            yield Utterance(
                text=text,
                pre_pause_ms=p.heading_pre.get(span.level, 400),
                post_pause_ms=p.heading_post.get(span.level, 250),
                role=f"heading.{span.level}",
            )

        elif isinstance(span, Paragraph):
            text = _norm(span.text)
            if not text: continue
            yield Utterance(text=text, post_pause_ms=p.paragraph_post, role="paragraph")

        elif isinstance(span, ListItem):
            text = _norm(span.text)
            if not text: continue
            yield Utterance(
                text=text,
                post_pause_ms=p.list_post if span.is_last else p.list_item_post,
                role="list.item",
            )

        elif isinstance(span, BlockQuote):
            text = _norm(span.text)
            if not text: continue
            yield Utterance(
                text=f"Quote. {text} End quote.",
                pre_pause_ms=p.blockquote_pre,
                post_pause_ms=p.blockquote_post,
                role="blockquote",
            )

        elif isinstance(span, CodeBlock):
            if p.skip_code and p.code_summary:
                lang = f" in {span.lang}" if span.lang else ""
                word = "line" if span.line_count == 1 else "lines"
                yield Utterance(
                    text=f"Code block{lang}, {span.line_count} {word}.",
                    pre_pause_ms=p.code_pre, post_pause_ms=p.code_post,
                    role="code.summary",
                )
            elif p.skip_code:
                yield Utterance(text="Code block.", pre_pause_ms=p.code_pre,
                                post_pause_ms=p.code_post, role="code.skip")
            else:
                yield Utterance(text=_norm(span.text), pre_pause_ms=p.code_pre,
                                post_pause_ms=p.code_post, role="code.verbatim")

        elif isinstance(span, Equation):
            # Route block equations through the detector pipeline by wrapping
            # in $...$ so whatever EquationDetector is configured handles them.
            spoken = _norm(f"${span.text}$")
            yield Utterance(
                text=spoken or "Equation.",
                pre_pause_ms=p.equation_pre, post_pause_ms=p.equation_post,
                role="equation",
            )

        elif isinstance(span, HorizontalRule):
            if p.skip_horizontal_rule_sound:
                yield Utterance(text="", pre_pause_ms=p.rule_pre,
                                post_pause_ms=p.rule_post, role="rule")
            else:
                yield Utterance(text="Section break.", pre_pause_ms=p.rule_pre,
                                post_pause_ms=p.rule_post, role="rule")

        elif isinstance(span, TableSummary):
            yield from _vocalize_table(span, p, _norm)


def _vocalize_table(span: TableSummary, p: Policy, normalize):
    """Render a TableSummary as one or more Utterances.

    - Headers + at most `table_max_rows_to_read` rows -> read row by row.
    - Otherwise: announce headers and dimensions only.
    """
    headers = [normalize(h) for h in span.headers]

    if 0 < span.n_rows <= p.table_max_rows_to_read and headers:
        # Per-row reading. We pair each cell with its header so the listener
        # can follow even when columns are reordered or sparse.
        n = len(headers)
        intro = f"Table with columns: {_join_phrase(headers)}."
        yield Utterance(
            text=intro,
            pre_pause_ms=p.table_pre,
            post_pause_ms=p.list_item_post,
            role="table.header",
        )
        for row_idx, row in enumerate(span.rows):
            cells = [normalize(c) for c in row]
            parts = []
            for j in range(min(n, len(cells))):
                if cells[j]:
                    parts.append(f"{headers[j]}: {cells[j]}")
            body = "; ".join(parts) + "." if parts else "(empty row)."
            # Establish the row pattern out loud on the first two rows.
            # After that the listener has the rhythm, so we go silent.
            if row_idx == 0:
                row_text = f"First row. {body}"
            elif row_idx == 1:
                row_text = f"Next row. {body}"
            else:
                row_text = body
            is_last = row_idx == len(span.rows) - 1
            yield Utterance(
                text=row_text,
                post_pause_ms=p.table_post if is_last else p.list_item_post,
                role="table.row",
            )
        return

    # Summary path.
    if headers:
        text = (
            f"Table with columns: {_join_phrase(headers)}. "
            f"{span.n_rows} {'row' if span.n_rows == 1 else 'rows'}."
        )
    else:
        text = f"Table with {span.n_rows} rows and {span.n_cols} columns."
    yield Utterance(
        text=text,
        pre_pause_ms=p.table_pre,
        post_pause_ms=p.table_post,
        role="table",
    )


def _join_phrase(items: list[str]) -> str:
    """Speakable join: ['a','b','c'] -> 'a, b, and c'."""
    items = [s for s in items if s]
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
