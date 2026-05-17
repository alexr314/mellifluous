"""Smoke tests for the markdown parser."""
from mellifluous.parse import parse_markdown
from mellifluous.spans import (
    Heading, Paragraph, ListItem, BlockQuote, CodeBlock,
    Equation, HorizontalRule, TableSummary,
)


def types(doc):
    return [type(s).__name__ for s in doc]


def test_basic_structure():
    md = "# Title\n\nA paragraph.\n\n## Sub\n\nMore text.\n"
    doc = parse_markdown(md)
    assert types(doc) == ["Heading", "Paragraph", "Heading", "Paragraph"]
    assert doc[0].level == 1
    assert doc[2].level == 2
    assert doc[1].text == "A paragraph."


def test_lists_have_position_info():
    md = "- a\n- b\n- c\n"
    doc = parse_markdown(md)
    assert types(doc) == ["ListItem", "ListItem", "ListItem"]
    assert doc[0].is_first and not doc[0].is_last
    assert not doc[1].is_first and not doc[1].is_last
    assert doc[2].is_last and not doc[2].is_first
    assert [s.index for s in doc] == [1, 2, 3]


def test_ordered_vs_unordered_lists():
    bullet = parse_markdown("- a\n- b\n")
    ordered = parse_markdown("1. a\n2. b\n")
    assert all(s.ordered is False for s in bullet)
    assert all(s.ordered is True for s in ordered)


def test_blockquote_is_collapsed_to_single_span():
    md = "> first paragraph\n>\n> second paragraph\n"
    doc = parse_markdown(md)
    assert len(doc) == 1
    assert isinstance(doc[0], BlockQuote)
    assert "first paragraph" in doc[0].text
    assert "second paragraph" in doc[0].text


def test_code_block_with_language():
    md = "```python\ndef f(): pass\n```\n"
    doc = parse_markdown(md)
    assert len(doc) == 1
    assert isinstance(doc[0], CodeBlock)
    assert doc[0].lang == "python"
    assert doc[0].line_count == 1


def test_horizontal_rule():
    doc = parse_markdown("A.\n\n---\n\nB.\n")
    assert types(doc) == ["Paragraph", "HorizontalRule", "Paragraph"]


def test_links_keep_text_drop_url():
    doc = parse_markdown("Visit [the docs](https://example.com/docs) please.")
    assert "the docs" in doc[0].text
    assert "https://example.com" not in doc[0].text


def test_dollar_inline_math_survives_as_dollar_wrapped():
    """Inline $...$ becomes math_inline tokens; we re-emit as $...$ so the
    EquationDetector can claim it downstream."""
    doc = parse_markdown(r"Einstein: $E = mc^2$.")
    assert "$E = mc^2$" in doc[0].text


def test_bracket_inline_math_survives():
    r"""\(...\) is the texmath plugin's brackets delimiter."""
    doc = parse_markdown(r"Pythagoras: \(a^2 + b^2 = c^2\).")
    assert "$a^2 + b^2 = c^2$" in doc[0].text


def test_display_math_becomes_equation_span():
    doc = parse_markdown("Intro.\n\n$$ E = mc^2 $$\n\nOutro.")
    assert any(isinstance(s, Equation) for s in doc)


def test_inline_code_kept_in_backticks_for_downstream_detector():
    doc = parse_markdown("Call `df.merge()` here.")
    assert "`df.merge()`" in doc[0].text


def test_table_summarized_not_read_verbatim():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    doc = parse_markdown(md)
    tables = [s for s in doc if isinstance(s, TableSummary)]
    assert len(tables) == 1
    assert tables[0].n_rows == 2
    assert tables[0].n_cols == 2
