"""Smoke tests for the vocalizer: Spans -> Utterances with pauses."""
from mellifluous.parse    import parse_markdown
from mellifluous.vocalize import vocalize, Policy


def test_heading_carries_larger_pre_pause_than_paragraph():
    utts = list(vocalize(parse_markdown("# Title\n\nBody.")))
    assert utts[0].role == "heading.1"
    assert utts[1].role == "paragraph"
    assert utts[0].pre_pause_ms > utts[1].pre_pause_ms


def test_list_items_keep_short_gaps_except_after_last():
    utts = list(vocalize(parse_markdown("- a\n- b\n- c\n")))
    roles = [u.role for u in utts]
    assert roles == ["list.item", "list.item", "list.item"]
    # The last item gets the longer "list closing" pause.
    assert utts[-1].post_pause_ms > utts[0].post_pause_ms


def test_block_equation_routed_through_detector_pipeline():
    """Regression: display $$...$$ used to short-circuit to 'Equation.'
    Now it goes through the EquationDetector so a plugged-in LLM reader
    handles display math too."""
    utts = list(vocalize(parse_markdown("$$ E = mc^2 $$")))
    eq = [u for u in utts if u.role == "equation"]
    assert len(eq) == 1
    # The default rule-based reader handles E = mc^2.
    assert "equals" in eq[0].text.lower()


def test_blockquote_wraps_text_in_quote_markers():
    utts = list(vocalize(parse_markdown("> Be kind to bots.")))
    q = next(u for u in utts if u.role == "blockquote")
    assert q.text.startswith("Quote.")
    assert q.text.endswith("End quote.")


def test_code_block_summarized_by_default():
    md = "```python\ndef f(): pass\n```\n"
    utts = list(vocalize(parse_markdown(md)))
    cb = next(u for u in utts if u.role.startswith("code"))
    assert cb.role == "code.summary"
    assert "python" in cb.text
    assert "1" in cb.text and "line" in cb.text


def test_table_default_announces_headers_and_row_count():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    utts = list(vocalize(parse_markdown(md)))
    t = next(u for u in utts if u.role == "table")
    assert "columns" in t.text and "A" in t.text and "B" in t.text
    assert "2 rows" in t.text


def test_table_reads_rows_when_enabled():
    md = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |\n"
    utts = list(vocalize(parse_markdown(md), policy=Policy(table_max_rows_to_read=5)))
    roles = [u.role for u in utts]
    assert "table.header" in roles
    assert roles.count("table.row") == 3
    rows = [u for u in utts if u.role == "table.row"]
    # First two rows are announced; subsequent rows are not.
    assert rows[0].text.startswith("First row.")
    assert rows[1].text.startswith("Next row.")
    assert not rows[2].text.startswith(("First", "Next"))
    # Header-paired cell content is still present.
    assert "A: 1" in rows[0].text and "B: 2" in rows[0].text


def test_table_falls_back_to_summary_when_too_many_rows():
    md = "| A |\n|---|\n" + "\n".join("| x |" for _ in range(20))
    utts = list(vocalize(parse_markdown(md), policy=Policy(table_max_rows_to_read=5)))
    roles = [u.role for u in utts]
    assert "table" in roles
    assert "table.row" not in roles


def test_policy_overrides_pauses():
    short = Policy(paragraph_post=10)
    long  = Policy(paragraph_post=900)
    a = list(vocalize(parse_markdown("Hello."), policy=short))[0]
    b = list(vocalize(parse_markdown("Hello."), policy=long))[0]
    assert a.post_pause_ms == 10
    assert b.post_pause_ms == 900
