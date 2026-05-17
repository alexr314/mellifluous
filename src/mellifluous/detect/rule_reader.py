"""A rule-based LaTeX-to-speech reader for EquationDetector.

This is the default no-LLM upgrade path: handles common shapes (=, +, -,
fractions, sqrt, sums, integrals, powers, subscripts, Greek letters) by
substitution, falls back to the word "equation" for anything it can't
clearly express.

The output is not as natural as an LLM reader, but it's free, offline, and
covers a meaningful subset of everyday math notation.
"""
from __future__ import annotations
import re


# Greek letters (reuses what EquationDetector already knows for trivials).
_GREEK = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\delta": "delta",
    r"\epsilon": "epsilon", r"\varepsilon": "epsilon", r"\zeta": "zeta",
    r"\eta": "eta", r"\theta": "theta", r"\vartheta": "theta",
    r"\iota": "iota", r"\kappa": "kappa", r"\lambda": "lambda", r"\mu": "mu",
    r"\nu": "nu", r"\xi": "xi", r"\pi": "pi", r"\rho": "rho", r"\varrho": "rho",
    r"\sigma": "sigma", r"\tau": "tau", r"\upsilon": "upsilon",
    r"\phi": "phi", r"\varphi": "phi", r"\chi": "chi", r"\psi": "psi", r"\omega": "omega",
    r"\Gamma": "capital gamma", r"\Delta": "capital delta", r"\Theta": "capital theta",
    r"\Lambda": "capital lambda", r"\Xi": "capital xi", r"\Pi": "capital pi",
    r"\Sigma": "capital sigma", r"\Phi": "capital phi", r"\Psi": "capital psi",
    r"\Omega": "capital omega",
    r"\infty": "infinity", r"\nabla": "nabla", r"\partial": "partial",
    r"\hbar": "h bar",
}


# Common operator-style commands. Spoken inline.
_OPS = {
    r"\times": "times", r"\cdot": "dot", r"\div": "divided by",
    r"\pm": "plus or minus", r"\mp": "minus or plus",
    r"\leq": "less than or equal to", r"\geq": "greater than or equal to",
    r"\neq": "not equal to", r"\approx": "approximately",
    r"\equiv": "equivalent to", r"\sim": "similar to",
    r"\to": "to", r"\rightarrow": "to", r"\Rightarrow": "implies",
    r"\leftarrow": "from", r"\in": "in", r"\notin": "not in",
    r"\subset": "subset of", r"\supset": "superset of",
    r"\cup": "union", r"\cap": "intersection",
}


# Function-like commands that consume one or two braced arguments.
def _read_braced(s: str, i: int) -> tuple[str, int] | None:
    """If s[i] == '{', return (contents, end_index_after_close). Else None."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 1
    j = i + 1
    while j < len(s) and depth > 0:
        if s[j] == "\\" and j + 1 < len(s):
            j += 2
            continue
        if s[j] == "{": depth += 1
        elif s[j] == "}": depth -= 1
        j += 1
    if depth != 0:
        return None
    return s[i + 1:j - 1], j


def _read_one_token(s: str, i: int) -> tuple[str, int]:
    """Read a single 'subscript token' starting at s[i].

    Handles three forms:
      {...}          braced group
      \\command      a backslash followed by letters (one LaTeX command token)
      X              any single non-space character

    Returns (token_text, index_after_token).
    """
    if i >= len(s):
        return "", i
    if s[i] == "{":
        a = _read_braced(s, i)
        if a: return a
        return s[i], i + 1
    if s[i] == "\\":
        # Consume backslash + letter run.
        j = i + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        if j == i + 1:                # \. or \, etc.
            j = i + 2 if j + 1 <= len(s) else i + 1
        return s[i:j], j
    return s[i], i + 1


def _strip_modifiers(latex: str) -> str:
    """Drop spacing and a few cosmetic LaTeX commands the listener doesn't need."""
    # spacing
    latex = re.sub(r"\\[,\;\!\:>\s]", " ", latex)
    # text formatting we want to ignore for speech
    latex = re.sub(r"\\(?:displaystyle|mathrm|operatorname|mathbf|mathit|text)\b", "", latex)
    # tfrac / dfrac behave like frac for speech
    latex = latex.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    return latex


def _has_unsupported(latex: str) -> bool:
    """Heuristic: if the remaining LaTeX still has commands we don't know,
    we'd rather say 'equation' than mangle them."""
    return bool(re.search(r"\\[A-Za-z]+", latex))


_BIGOPS = [("sum", "sum"), ("prod", "product"), ("int", "integral")]


def _expand_bigop(s: str, latex_cmd: str, word: str) -> str:
    """Expand \\sum / \\prod / \\int with optional _lower and ^upper bounds.

      \\sum_{i=1}^{n}   ->  "sum from i equals 1 to n of"
      \\int_a^b         ->  "integral from a to b of"
      \\prod_{k=0}      ->  "product over k equals 0 of"
    """
    cmd = "\\" + latex_cmd
    out = []
    i = 0
    while i < len(s):
        k = s.find(cmd, i)
        if k < 0:
            out.append(s[i:]); break
        # Word-boundary: don't match \sum inside \summon.
        end_of_cmd = k + len(cmd)
        if end_of_cmd < len(s) and s[end_of_cmd].isalpha():
            out.append(s[i:end_of_cmd]); i = end_of_cmd; continue
        out.append(s[i:k])
        j = end_of_cmd
        while j < len(s) and s[j].isspace(): j += 1
        lower = upper = None
        if j < len(s) and s[j] == "_":
            lower, j = _read_one_token(s, j + 1)
        while j < len(s) and s[j].isspace(): j += 1
        if j < len(s) and s[j] == "^":
            upper, j = _read_one_token(s, j + 1)
        if lower and upper:
            phrase = f"{word} from {rule_based_reader(lower)} to {rule_based_reader(upper)} of"
        elif lower:
            phrase = f"{word} over {rule_based_reader(lower)} of"
        else:
            phrase = word
        out.append(f" {phrase} ")
        i = j
    return "".join(out)


def rule_based_reader(latex: str) -> str:
    """Return a spoken English rendering, or 'equation' if too complex."""
    s = _strip_modifiers(latex).strip()

    # Bigops (\sum, \prod, \int) must run before any Greek substitution,
    # because their bounds may contain commands like \infty that we want
    # to recursively render correctly inside the bound, not after it's
    # been stranded as a partial substring.
    for cmd, word in _BIGOPS:
        s = _expand_bigop(s, cmd, word)

    # Greek + named constants. Lookahead prevents partial matches: without it,
    # replacing \in -> "in" would mangle \int and \infty.
    for tex, word in _GREEK.items():
        s = re.sub(re.escape(tex) + r"(?![A-Za-z])", word, s)
    for tex, word in _OPS.items():
        s = re.sub(re.escape(tex) + r"(?![A-Za-z])", f" {word} ", s)

    # \frac{a}{b}  ->  "a over b"
    def _expand_two_arg(name: str, joiner: str) -> str:
        nonlocal s
        out = []
        i = 0
        cmd = "\\" + name
        while i < len(s):
            k = s.find(cmd, i)
            if k < 0:
                out.append(s[i:])
                break
            out.append(s[i:k])
            j = k + len(cmd)
            a = _read_braced(s, j)
            if not a:
                out.append(s[k:k + len(cmd)]); i = k + len(cmd); continue
            b = _read_braced(s, a[1])
            if not b:
                out.append(s[k:k + len(cmd)]); i = k + len(cmd); continue
            out.append(f" {rule_based_reader(a[0])} {joiner} {rule_based_reader(b[0])} ")
            i = b[1]
        return "".join(out)

    s = _expand_two_arg("frac", "over")

    # \sqrt{x} -> "square root of x"
    def _expand_one_arg(name: str, fmt: str) -> str:
        nonlocal s
        out = []
        i = 0
        cmd = "\\" + name
        while i < len(s):
            k = s.find(cmd, i)
            if k < 0:
                out.append(s[i:]); break
            out.append(s[i:k])
            j = k + len(cmd)
            a = _read_braced(s, j)
            if not a:
                out.append(s[k:k + len(cmd)]); i = k + len(cmd); continue
            out.append(f" {fmt.format(arg=rule_based_reader(a[0]))} ")
            i = a[1]
        return "".join(out)

    s = _expand_one_arg("sqrt", "square root of {arg}")
    s = _expand_one_arg("hat",  "{arg} hat")
    s = _expand_one_arg("bar",  "{arg} bar")
    s = _expand_one_arg("vec",  "vector {arg}")
    s = _expand_one_arg("tilde","{arg} tilde")
    s = _expand_one_arg("dot",  "{arg} dot")

    # Superscripts: x^2 -> "x squared", x^3 -> "x cubed", x^{n} -> "x to the n"
    POWER_WORDS = {"2": "squared", "3": "cubed"}
    def _power_sub(m: re.Match) -> str:
        exp = m.group(1) or m.group(2) or ""
        if exp in POWER_WORDS:
            return " " + POWER_WORDS[exp]
        return f" to the {rule_based_reader(exp)}" if exp else ""
    s = re.sub(r"\^\{([^{}]*)\}|\^([A-Za-z0-9])", _power_sub, s)

    # Subscripts: x_1 -> "x sub 1", x_{i+1} -> "x sub i plus 1"
    def _sub_sub(m: re.Match) -> str:
        v = m.group(1) or m.group(2) or ""
        return f" sub {rule_based_reader(v)}" if v else ""
    s = re.sub(r"_\{([^{}]*)\}|_([A-Za-z0-9])", _sub_sub, s)

    # Bare ASCII operators that the symbol detector also handles.
    s = (s.replace("=", " equals ")
           .replace("+", " plus ")
           .replace("-", " minus ")
           .replace("/", " over ")
           .replace("*", " times "))

    # Cleanup: collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()

    # If anything LaTeX-y is still in the string, we missed something. Be honest.
    if _has_unsupported(s) or "{" in s or "}" in s:
        return "equation"
    return s
