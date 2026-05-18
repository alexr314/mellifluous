"""Smoke tests for the detector pipeline."""
import pytest

from mellifluous.detect import (
    Pipeline, Claimed, Unclaimed, normalize_text,
    EquationDetector, UrlDetector, InlineCodeDetector,
    NumberDetector, SymbolDetector,
    default_pipeline,
)


# --- core pipeline invariants ------------------------------------------------

def test_pipeline_preserves_raw_text():
    p = default_pipeline()
    src = "Visit https://example.com for 5% off. Use `df.merge()`. $E = mc^2$."
    segments = p.process(src)
    assert "".join(s.raw for s in segments) == src


def test_pipeline_runs_in_priority_order():
    """URLs are claimed before NumberDetector so a URL with digits doesn't
    get its numbers mangled."""
    p = default_pipeline()
    spoken = normalize_text(p, "See https://example.com/v2 for 5% off")
    # The URL collapses to the domain; the 5% becomes '5 percent'. The v2
    # path digits stay inside the URL claim and aren't number-detected.
    assert "example dot com" in spoken
    assert "5 percent" in spoken


def test_detector_that_breaks_invariant_raises():
    class Bad:
        name = "bad"
        priority = 5
        def scan(self, text):
            return [Claimed(raw="x", spoken="x", detector=self.name)]

    with pytest.raises(RuntimeError, match="did not preserve text"):
        Pipeline([Bad()]).process("hello")


# --- equation detector -------------------------------------------------------

def test_equation_default_reader_handles_simple_cases():
    """Default reader is rule-based: handles common shapes, says 'equation'
    only when the expression is too complex to parse."""
    p = Pipeline([EquationDetector()])
    # Simple case the rule reader can handle.
    assert "E equals" in normalize_text(p, "We have $E = mc^2$.")
    # Pathological case it can't parse.
    out = normalize_text(p, r"Mess: $\mysteryop{x}$.")
    assert "equation" in out


def test_equation_trivial_fast_path_skips_reader():
    """Single-letter variables shouldn't burn an LLM call."""
    calls = []
    p = Pipeline([EquationDetector(reader=lambda x: calls.append(x) or "FAIL")])
    out = normalize_text(p, "The variable $n$ in $\\alpha$ position.")
    assert calls == [], "reader was called for trivial expressions"
    assert "n" in out and "alpha" in out


def test_equation_recognizes_all_four_delimiters():
    seen = []
    p = Pipeline([EquationDetector(reader=lambda x: seen.append(x) or "EQ")])
    samples = [
        "inline $a + b$ here",
        r"bracket \(c + d\) here",
        "display $$e + f$$ here",
        r"bracket display \[g + h\] here",
    ]
    for s in samples:
        normalize_text(p, s)
    assert seen == ["a + b", "c + d", "e + f", "g + h"]


def test_inline_currency_not_mistaken_for_equation():
    """$5 is a price, not a math expression."""
    p = Pipeline([EquationDetector(reader=lambda x: "EQ")])
    assert "EQ" not in normalize_text(p, "It costs $5 today.")


# --- URL ---------------------------------------------------------------------

def test_url_speaks_domain_drops_path():
    p = Pipeline([UrlDetector()])
    out = normalize_text(p, "See https://example.com/path/x?q=1 here.")
    assert "example dot com" in out
    assert "path" not in out


def test_email_detected():
    p = Pipeline([UrlDetector()])
    out = normalize_text(p, "Mail alex@example.com please.")
    assert "alex at example dot com" in out


def test_email_local_part_with_dot_is_spoken():
    p = Pipeline([UrlDetector()])
    out = normalize_text(p, "Write to first.last@example.com.")
    assert "first dot last at example dot com" in out


def test_url_phrasing_does_not_double_up():
    """Regression: 'email alex@example.com' used to read 'email email at example...'.
    With the cleaner phrasing the surrounding 'email' word stays put."""
    p = Pipeline([UrlDetector()])
    out = normalize_text(p, "or email alex@example.com.")
    assert out.count("email") == 1
    assert "example dot com" in out


# --- inline code -------------------------------------------------------------

def test_inline_code_strips_punctuation():
    p = Pipeline([InlineCodeDetector()])
    out = normalize_text(p, "Run `df.merge(left, right)` now.")
    assert "df merge left right" in out


def test_inline_code_speaks_operators_as_words():
    """`a == b && c != d` should read like English, not symbols."""
    p = Pipeline([InlineCodeDetector()])
    out = normalize_text(p, "The check `a == b && c != d` passes.")
    assert "equals" in out
    assert "not equal" in out
    assert "and" in out
    assert "==" not in out
    assert "&&" not in out


# --- numbers -----------------------------------------------------------------

def test_dollar_amounts_dont_eat_trailing_punctuation():
    """Regression: $1,200, used to become '1,200, dollars'."""
    p = Pipeline([NumberDetector()])
    out = normalize_text(p, "It cost $1,200, plus tax.")
    assert "1,200 dollars" in out
    assert "1,200,  dollars" not in out
    assert "1,200, dollars" not in out


def test_dollar_amounts_treat_decimals_as_cents():
    """$1,000.50 should read 'and 50 cents', not 'point fifty dollars'."""
    p = Pipeline([NumberDetector()])
    out = normalize_text(p, "$1,000.50 today.")
    assert "1,000 dollars and 50 cents" in out


def test_dollar_one_digit_after_point_means_tens_of_cents():
    """$1,000.5 means 50 cents, not 5 cents."""
    p = Pipeline([NumberDetector()])
    out = normalize_text(p, "$1,000.5 received.")
    assert "1,000 dollars and 50 cents" in out


def test_dollar_under_a_dollar_speaks_only_cents():
    p = Pipeline([NumberDetector()])
    out = normalize_text(p, "It costs $0.99 today.")
    assert "99 cents" in out
    assert "dollars" not in out


def test_percent_and_units():
    p = Pipeline([NumberDetector()])
    out = normalize_text(p, "Grew 25% to 10kg.")
    assert "25 percent" in out
    assert "10 kg" in out


# --- symbols -----------------------------------------------------------------

def test_arrows_and_operators():
    p = Pipeline([SymbolDetector()])
    out = normalize_text(p, "if a==b && c!=d then x -> y")
    assert "equals" in out
    assert "and" in out
    assert "not equal" in out
    assert "arrow" in out


# --- end-to-end through default pipeline ------------------------------------

def test_full_pipeline_on_mixed_content():
    p = default_pipeline()
    src = "Buy at https://shop.example.com for $1,200, save 25%. `apply()` then x->y."
    out = normalize_text(p, src)
    assert "shop dot example dot com" in out
    assert "1,200 dollars" in out
    assert "25 percent" in out
    assert "apply" in out
    assert "arrow" in out
