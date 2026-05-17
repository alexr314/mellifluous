"""The sentence splitter is used by the Streamer when fed an iterator of
strings (LLM token streams). Verify it handles the edge cases that
broke earlier prototypes."""
from mellifluous.synthesize.streamer import _drain_sentences


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
