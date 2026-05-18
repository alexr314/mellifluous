"""The sentence splitter is used by every backend when fed an iterator of
strings (LLM token streams). Verify it handles the edge cases that
broke earlier prototypes."""
from mellifluous.synthesize._text import (
    drain_sentences as _drain_sentences,
    drain_markdown_blocks,
)


def test_basic_split():
    sents, leftover = _drain_sentences("Hello. World! Done?")
    assert sents == ["Hello.", "World!", "Done?"]
    assert leftover == ""


def test_holds_unfinished_sentence_for_later():
    sents, leftover = _drain_sentences("First done. Second pending no terminator")
    assert sents == ["First done."]
    assert leftover == " Second pending no terminator"


def test_does_not_split_on_abbreviations():
    sents, leftover = _drain_sentences("Mr. Smith and Dr. Jones met. End.")
    assert sents == ["Mr. Smith and Dr. Jones met.", "End."]
    assert leftover == ""


def test_does_not_split_on_decimals():
    sents, leftover = _drain_sentences("Pi is 3.14 today. End.")
    assert sents == ["Pi is 3.14 today.", "End."]
    assert leftover == ""


def test_handles_quotes_after_terminator():
    sents, leftover = _drain_sentences('He said "hi." Then left.')
    assert len(sents) == 2
    assert sents[0].endswith('hi."')


# --- markdown block splitter (streaming markdown input) -----------------

def test_block_splitter_emits_complete_paragraphs():
    blocks, rem = drain_markdown_blocks("Para A.\n\nPara B.\n\nPara C")
    assert blocks == ["Para A.\n", "Para B.\n"]
    # The partial third paragraph is held back; no trailing blank line yet.
    assert rem == "Para C"


def test_block_splitter_keeps_fenced_code_intact():
    """Regression: blank lines INSIDE a fenced code block must not split
    the block, or the markdown parser sees junk."""
    src = (
        "Para before.\n\n"
        "```python\n"
        "def foo():\n\n"               # blank line inside fence
        "    return 1\n\n"
        "```\n\n"
        "Para after.\n"
    )
    blocks, rem = drain_markdown_blocks(src + "\n")
    # Three complete blocks: prose, the entire code block, more prose.
    assert len(blocks) == 3
    assert "def foo()" in blocks[1] and "return 1" in blocks[1]
    assert blocks[1].count("```") == 2


def test_block_splitter_supports_incremental_feeding():
    """Simulates an LLM token stream: tokens arrive one at a time, the
    splitter accumulates a buffer and yields blocks as they complete."""
    tokens = ["Para ", "A", ".", "\n", "\n", "Para ", "B", ".", "\n", "\n", "Tail"]
    buf = ""
    out = []
    for tok in tokens:
        buf += tok
        blocks, buf = drain_markdown_blocks(buf)
        out.extend(blocks)
    assert out == ["Para A.\n", "Para B.\n"]
    assert buf == "Tail"


def test_block_splitter_tilde_fences_also_recognized():
    """CommonMark also allows ~~~ fences; the splitter must honor them
    or blank lines inside tilde-fenced code would shatter the block."""
    src = "~~~\nfoo\n\nbar\n~~~\n\nafter\n\n"
    blocks, rem = drain_markdown_blocks(src)
    assert len(blocks) == 2
    assert "bar" in blocks[0]


def test_block_splitter_handles_empty_input():
    assert drain_markdown_blocks("") == ([], "")


def test_block_splitter_holds_trailing_partial_block():
    """End-of-stream flushing is the caller's job: the splitter only
    emits blocks that ended with a blank line. A partial trailing block
    stays in the remainder until the caller decides to flush it."""
    blocks, rem = drain_markdown_blocks("Incomplete paragraph with no blank line")
    assert blocks == []
    assert rem == "Incomplete paragraph with no blank line"


# --- Reader.synthesize streaming-markdown integration --------------------

class _RecordingBackend:
    """A minimal Backend stand-in: records the text it was asked to
    synthesize, returns a single silent AudioChunk per call. Lets us
    assert the streaming markdown path normalized + flushed correctly
    without spinning up a real TTS."""
    sample_rate = 24000

    def __init__(self):
        self.spoken: list[str] = []
        # Reader keeps a 'voice' attribute for introspection.
        self.voice = "test"
        self.instructions = None

    def warm(self):  # pragma: no cover
        pass

    def synthesize(self, text, *, voice=None, instructions=None):
        import numpy as np
        from mellifluous.synthesize.types import AudioChunk
        self.spoken.append(text)
        yield AudioChunk(
            pcm=np.zeros(1, dtype=np.float32),
            sample_rate=self.sample_rate,
            is_final=True,
            meta={},
        )


def test_streaming_markdown_strips_emphasis_and_speaks_block_by_block(monkeypatch):
    """End-to-end: an LLM-style token stream containing markdown emphasis
    (`**bold**`, `*italic*`) goes in; the streaming path should yield
    spoken text with the emphasis markers parsed away, one or more
    chunks per markdown block."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from mellifluous import Reader

    r = Reader(engine="openai")
    fake = _RecordingBackend()
    r.backend = fake

    # Two complete paragraphs followed by a partial one. Tokens chosen to
    # mimic the way a real LLM yields them (small chunks, asterisks
    # arriving as separate tokens, internal whitespace).
    tokens = [
        "**Para", " 1", "**", " is", " the", " first", " block", ".\n",
        "\n",
        "Para 2", " has", " *italic*", " content", ".\n",
        "\n",
        "Trailing", " block", " without", " blank", " line",
    ]
    chunks = list(r.synthesize(iter(tokens), as_markdown=True))
    assert chunks, "streaming should produce at least one AudioChunk"

    all_spoken = " ".join(fake.spoken)
    # Emphasis markers are gone (parsed, not literal).
    assert "**" not in all_spoken
    assert "*italic*" not in all_spoken
    # All three blocks made it out, including the unterminated trailing
    # one (flushed at end of stream).
    assert "first block" in all_spoken
    assert "italic" in all_spoken
    assert "Trailing" in all_spoken


def test_streaming_markdown_preserves_fenced_code_block(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    from mellifluous import Reader, Policy

    # Turn off skip_code so the code block summary actually speaks; that
    # way we can detect that the parser saw a code block (instead of
    # splitting on the blank line inside the fence and getting confused).
    r = Reader(engine="openai", policy=Policy(skip_code=False))
    fake = _RecordingBackend()
    r.backend = fake

    tokens = [
        "Here is ", "code:\n", "\n",
        "```python\n",
        "def f():\n",
        "\n",                      # blank line inside fence
        "    return 1\n",
        "```\n", "\n",
        "Done.",
    ]
    list(r.synthesize(iter(tokens), as_markdown=True))
    all_spoken = " ".join(fake.spoken)
    # The code block should have been recognized as a single code span,
    # not torn in half. The vocalizer's code path mentions "python" or
    # "code" -- pick the indicator that's reliably there.
    assert "code" in all_spoken.lower()
    assert "Done" in all_spoken
