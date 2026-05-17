"""Tests for the rule-based LaTeX-to-speech fallback reader."""
from mellifluous.detect.rule_reader import rule_based_reader as r


def test_simple_equation():
    assert r("E = mc^2") == "E equals mc squared"


def test_fraction():
    assert "over" in r(r"\frac{a}{b}")
    assert "1 over 2" in r(r"\frac{1}{2}")


def test_sqrt():
    out = r(r"\sqrt{2}")
    assert "square root of 2" in out


def test_subscripts_and_superscripts():
    assert "sub 1" in r("x_1")
    assert "to the n" in r("x^n")
    assert "squared" in r("x^2")
    assert "cubed" in r("x^3")


def test_greek():
    assert "alpha" in r(r"\alpha + \beta")
    assert "pi" in r(r"\pi")


def test_named_operators():
    assert "less than or equal to" in r(r"a \leq b")
    assert "infinity" in r(r"\infty")


def test_sum_with_bounds():
    out = r(r"\sum_{i=1}^{n} i^2")
    assert "sum from" in out
    assert "to" in out
    assert "squared" in out


def test_integral_with_bounds():
    out = r(r"\int_a^b f(x) dx")
    assert "integral from a to b" in out


def test_unparseable_falls_back_to_equation():
    """When we see an unsupported \\command, return 'equation' rather than
    mangling it."""
    assert r(r"\mysteriouscommand{x}") == "equation"


def test_spacing_macros_are_ignored():
    """\\, and friends are display spacing; speech doesn't need them."""
    out = r(r"a \, + \, b")
    # No literal backslash-comma left in the output.
    assert "\\" not in out
