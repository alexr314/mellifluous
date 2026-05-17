"""Markdown -> Document.

Uses markdown-it-py (CommonMark) with dollarmath + texmath plugins so we get
proper math tokens for `$...$`, `$$...$$`, `\\(...\\)`, and `\\[...\\]`.

Strategy: walk the flat token stream once with a small state machine. Block
open/close pairs (paragraph_open / heading_open / list_item_open / ...) push
frames; matching closes pop and emit a Span.
"""
from __future__ import annotations

from markdown_it import MarkdownIt
from mdit_py_plugins.texmath import texmath_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin

from .spans import (
    Span, Paragraph, Heading, ListItem, BlockQuote,
    CodeBlock, Equation, HorizontalRule, TableSummary, Document,
)


def _inline_text(token) -> str:
    """Collect speakable text from an inline token's children."""
    out = []
    for child in (token.children or []):
        t = child.type
        if t == "text":
            out.append(child.content)
        elif t == "code_inline":
            out.append(f"`{child.content}`")           # detector can rewrite later
        elif t == "math_inline":
            out.append(f"${child.content}$")           # detector will claim it
        elif t == "softbreak" or t == "hardbreak":
            out.append(" ")
        elif t in ("link_open", "link_close"):
            pass                                       # keep just the link text
        elif t == "image":
            alt = child.attrGet("alt") or ""
            if alt:
                out.append(f"image: {alt}")
        elif t == "html_inline":
            pass
        elif t.endswith("_open") or t.endswith("_close"):
            pass
        else:
            if getattr(child, "content", ""):
                out.append(child.content)
    return "".join(out).strip()


def _count_items(tokens, list_open_idx: int) -> int:
    depth = 0
    n = 0
    open_ty  = tokens[list_open_idx].type
    close_ty = open_ty.replace("_open", "_close")
    for k in range(list_open_idx + 1, len(tokens)):
        ty = tokens[k].type
        if ty == open_ty: depth += 1
        elif ty == close_ty:
            if depth == 0: return n
            depth -= 1
        elif depth == 0 and ty == "list_item_open":
            n += 1
    return n


def _summarize_table(tokens, table_open_idx: int):
    """Return (n_rows, n_cols, headers, rows).

    n_rows counts body rows only (the header row is broken out separately).
    Cell text is the speakable inline content (children flattened the same
    way as paragraph inline text).
    """
    headers: list[str] = []
    rows: list[list[str]] = []
    current_row: list[str] = []
    in_header = False
    in_row = False
    pending_cell_is_header = False

    k = table_open_idx + 1
    while k < len(tokens):
        ty = tokens[k].type
        if ty == "table_close":
            break
        if ty == "thead_open":
            in_header = True
        elif ty == "thead_close":
            in_header = False
        elif ty == "tr_open":
            in_row = True
            current_row = []
        elif ty == "tr_close":
            in_row = False
            if in_header:
                headers = current_row
            else:
                rows.append(current_row)
        elif in_row and ty in ("td_open", "th_open"):
            pending_cell_is_header = (ty == "th_open")
            # Next inline token is the cell content.
            # Find it and consume.
            j = k + 1
            cell_text = ""
            while j < len(tokens) and tokens[j].type not in ("td_close", "th_close"):
                if tokens[j].type == "inline":
                    cell_text = _inline_text(tokens[j])
                j += 1
            current_row.append(cell_text)
            k = j           # skip past the cell content + close
        k += 1

    n_cols = max((len(headers), *(len(r) for r in rows)), default=0)
    return len(rows), n_cols, tuple(headers), tuple(tuple(r) for r in rows)


def parse_markdown(src: str) -> Document:
    md = (
        MarkdownIt("commonmark", {"breaks": False, "html": False})
        .enable("table")                               # GFM-style pipe tables
        .use(dollarmath_plugin, allow_labels=True, double_inline=False)
        .use(texmath_plugin, delimiters="brackets")    # \(...\) and \[...\]
    )
    tokens = md.parse(src)

    out: list[Span] = []
    stack: list[tuple[str, dict]] = []
    inline_text: list[str] = []
    list_stack: list[dict] = []

    i = 0
    while i < len(tokens):
        t = tokens[i]
        ty = t.type

        if ty == "heading_open":
            stack.append(("heading", {"level": int(t.tag[1])}))
            inline_text.clear()
        elif ty == "heading_close":
            _, info = stack.pop()
            out.append(Heading(text=" ".join("".join(inline_text).split()), level=info["level"]))
            inline_text.clear()

        elif ty == "paragraph_open":
            stack.append(("paragraph", {}))
            inline_text.clear()
        elif ty == "paragraph_close":
            stack.pop()
            text = " ".join("".join(inline_text).split())
            inline_text.clear()
            if not text:
                pass
            elif stack and stack[-1][0] == "list_item":
                stack[-1][1].setdefault("text_parts", []).append(text)
            elif stack and stack[-1][0] == "blockquote":
                stack[-1][1].setdefault("text_parts", []).append(text)
            else:
                out.append(Paragraph(text=text))

        elif ty == "inline":
            inline_text.append(_inline_text(t))

        elif ty == "bullet_list_open" or ty == "ordered_list_open":
            list_stack.append({
                "ordered": ty == "ordered_list_open",
                "items_emitted": 0,
                "items_total":   _count_items(tokens, i),
            })
        elif ty == "bullet_list_close" or ty == "ordered_list_close":
            list_stack.pop()

        elif ty == "list_item_open":
            ls = list_stack[-1]
            ls["items_emitted"] += 1
            idx = ls["items_emitted"]
            stack.append(("list_item", {
                "ordered": ls["ordered"],
                "index": idx,
                "is_first": idx == 1,
                "is_last":  idx == ls["items_total"],
                "text_parts": [],
            }))
        elif ty == "list_item_close":
            _, info = stack.pop()
            text = " ".join(info.get("text_parts", []))
            if text:
                out.append(ListItem(
                    text=text,
                    ordered=info["ordered"],
                    index=info["index"],
                    is_first=info["is_first"],
                    is_last=info["is_last"],
                ))

        elif ty == "blockquote_open":
            stack.append(("blockquote", {"text_parts": []}))
        elif ty == "blockquote_close":
            _, info = stack.pop()
            text = " ".join(info.get("text_parts", []))
            if text:
                out.append(BlockQuote(text=text))

        elif ty in ("fence", "code_block"):
            content = t.content.rstrip("\n")
            out.append(CodeBlock(
                text=content,
                lang=(t.info or "").strip().split()[0] if t.info else None,
                line_count=content.count("\n") + 1 if content else 0,
            ))

        elif ty == "hr":
            out.append(HorizontalRule(text=""))

        elif ty == "table_open":
            n_rows, n_cols, headers, rows = _summarize_table(tokens, i)
            out.append(TableSummary(
                text="", n_rows=n_rows, n_cols=n_cols,
                headers=headers, rows=rows,
            ))
            while i < len(tokens) and tokens[i].type != "table_close":
                i += 1

        elif ty == "math_block":
            out.append(Equation(text=t.content.strip()))

        i += 1

    return out
