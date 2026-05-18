"""Built-in detectors: equations, URLs, inline code, numbers, symbols."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable

from .types import Segment, Claimed, Unclaimed


# ---------- helpers ----------

def _scan_regex(text: str, pattern: re.Pattern, speak: Callable[[re.Match], str], name: str) -> list[Segment]:
    out: list[Segment] = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            out.append(Unclaimed(text[last:m.start()]))
        out.append(Claimed(raw=m.group(0), spoken=speak(m), detector=name))
        last = m.end()
    if last < len(text):
        out.append(Unclaimed(text[last:]))
    return out or [Unclaimed(text)]


# ---------- equation ----------

_EQ_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)"),
    # Inline $...$. Non-greedy, requires non-space after opening $ so plain
    # currency amounts ("$5") don't get captured. NumberDetector handles those.
    re.compile(r"(?<![\w])\$(?!\s)([^\$\n]+?)(?<!\s)\$(?![\w])"),
]


def _default_equation_reader(latex: str) -> str:
    """Rule-based fallback. Handles common shapes; says 'equation' otherwise."""
    from .rule_reader import rule_based_reader
    return rule_based_reader(latex)


_TRIVIAL_VAR    = re.compile(r"^[a-zA-Z]$")
_TRIVIAL_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_GREEK = {
    r"\alpha": "alpha", r"\beta": "beta", r"\gamma": "gamma", r"\delta": "delta",
    r"\epsilon": "epsilon", r"\zeta": "zeta", r"\eta": "eta", r"\theta": "theta",
    r"\iota": "iota", r"\kappa": "kappa", r"\lambda": "lambda", r"\mu": "mu",
    r"\nu": "nu", r"\xi": "xi", r"\pi": "pi", r"\rho": "rho",
    r"\sigma": "sigma", r"\tau": "tau", r"\phi": "phi", r"\chi": "chi",
    r"\psi": "psi", r"\omega": "omega",
}

def _is_trivial(latex: str) -> bool:
    return bool(_TRIVIAL_VAR.match(latex) or _TRIVIAL_NUMBER.match(latex) or latex in _GREEK)

def _trivial_speak(latex: str) -> str:
    return _GREEK.get(latex, latex)


@dataclass
class EquationDetector:
    """Recognizes LaTeX math and routes it through a pluggable reader.

    The default reader returns "equation". Plug an LLM-backed one in for
    natural readings (see examples/groq_equation_reader.py).
    """
    reader: Callable[[str], str] = field(default=_default_equation_reader)
    name: str = "equation"
    priority: int = 10
    prefix: str = ""
    suffix: str = ""

    def scan(self, text: str) -> list[Segment]:
        out: list[Segment] = []
        i = 0
        while i < len(text):
            best = None
            for pat in _EQ_PATTERNS:
                m = pat.search(text, i)
                if m and (best is None or m.start() < best.start()):
                    best = m
            if best is None:
                out.append(Unclaimed(text[i:]))
                break
            if best.start() > i:
                out.append(Unclaimed(text[i:best.start()]))
            latex = best.group(1).strip()
            if _is_trivial(latex):
                spoken = _trivial_speak(latex)
            else:
                try:
                    spoken = self.reader(latex)
                except Exception:
                    spoken = "equation"
            if self.prefix or self.suffix:
                spoken = f"{self.prefix}{spoken}{self.suffix}"
            out.append(Claimed(raw=best.group(0), spoken=spoken, detector=self.name))
            i = best.end()
        return out or [Unclaimed(text)]


# ---------- URL ----------

_URL      = re.compile(r"https?://([^\s)>\]]+)", re.IGNORECASE)
_BARE_WWW = re.compile(r"\bwww\.([^\s)>\]]+)", re.IGNORECASE)
_EMAIL    = re.compile(r"\b([\w.+-]+)@([\w.-]+\.[A-Za-z]{2,})\b")


def _spoken_domain(host: str) -> str:
    host = host.split("/")[0].split("?")[0].strip(".,;:!?")
    return host.replace(".", " dot ")


def _spoken_email(local: str, domain: str) -> str:
    # Local part may have dots/pluses/hyphens; speak dots and read the rest
    # as-is. The TTS handles common names; unusual locals will sound spelled-
    # out but at least the listener gets the address.
    return f"{local.replace('.', ' dot ')} at {_spoken_domain(domain)}"


@dataclass
class UrlDetector:
    name: str = "url"
    priority: int = 20

    def scan(self, text: str) -> list[Segment]:
        # We say just the domain, no "link to" prefix. Markdown links like
        # [docs](https://...) are already collapsed to "docs" by the parser,
        # so a URL reaching us is bare and the listener already has context.
        segs = _scan_regex(text, _URL,
                           lambda m: _spoken_domain(m.group(1)), self.name)
        for pat, fmt in [
            (_BARE_WWW, lambda m: _spoken_domain(m.group(1))),
            (_EMAIL,    lambda m: _spoken_email(m.group(1), m.group(2))),
        ]:
            new: list[Segment] = []
            for s in segs:
                if isinstance(s, Claimed):
                    new.append(s); continue
                new.extend(_scan_regex(s.raw, pat, fmt, self.name))
            segs = new
        return segs


# ---------- inline code ----------

_INLINE_CODE = re.compile(r"`([^`]+)`")

def _speak_code(snippet: str) -> str:
    # Apply the operator-to-word substitutions first so things like `a==b`
    # become "a equals b" before we strip syntax noise. We share the table
    # with SymbolDetector; declared at the bottom of this module.
    s = snippet
    for pat, repl in _SYMBOL_TABLE:
        s = re.sub(pat, repl, s)
    s = s.replace(".", " ").replace("_", " ")
    s = re.sub(r"[(){}\[\];,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class InlineCodeDetector:
    name: str = "code"
    priority: int = 30

    def scan(self, text: str) -> list[Segment]:
        return _scan_regex(text, _INLINE_CODE,
                           lambda m: _speak_code(m.group(1)), self.name)


# ---------- dates ----------

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_ORDINALS = (
    "zeroth", "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
    "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
    "twenty-first", "twenty-second", "twenty-third", "twenty-fourth",
    "twenty-fifth", "twenty-sixth", "twenty-seventh", "twenty-eighth",
    "twenty-ninth", "thirtieth", "thirty-first",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty",
         "sixty", "seventy", "eighty", "ninety")
_LOW_TWO_DIGIT = (
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)


def _two_digit_words(n: int) -> str:
    """0..99 -> English words. Returns '' for 0 so callers can skip it."""
    if n < 20:
        return _LOW_TWO_DIGIT[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_LOW_TWO_DIGIT[ones]}"


def _year_to_words(year: int) -> str:
    """Years 1000..2099 spoken naturally:
    1999 -> 'nineteen ninety-nine', 2000 -> 'two thousand',
    2010 -> 'twenty ten', 2026 -> 'twenty twenty-six'.
    Years outside that range fall back to digit-by-digit reading.
    """
    if not (1000 <= year <= 2099):
        return " ".join(_LOW_TWO_DIGIT[int(d)] if d != "0" else "zero"
                        for d in str(year))
    century, rest = divmod(year, 100)
    if rest == 0:
        # 2000 -> "two thousand", 1900 -> "nineteen hundred"
        if century == 20:
            return "two thousand"
        return f"{_LOW_TWO_DIGIT[century % 100] or _two_digit_words(century)} hundred"
    if century == 20 and rest < 10:
        # 2001..2009 -> "two thousand one" ... "two thousand nine"
        return f"two thousand {_LOW_TWO_DIGIT[rest]}"
    # 1999 -> "nineteen ninety-nine"; 2026 -> "twenty twenty-six"
    return f"{_two_digit_words(century)} {_two_digit_words(rest)}"


def _spoken_date(year: int, month: int, day: int) -> str | None:
    """Return spoken form or None if any field is out of range."""
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{_MONTHS[month - 1]} {_ORDINALS[day]}, {_year_to_words(year)}"


# ISO date: 2026-05-17. Anchored on word boundaries so it doesn't capture
# pieces of longer numeric strings.
_DATE_ISO   = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
# US slash date: 05/17/2026 or 5/17/26. We assume US ordering (month/day/year)
# since that matches what shows up in the project's audit documents and on
# US-formatted text. Non-US documents would need their own detector.
_DATE_US    = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b")


def _speak_iso_date(m: re.Match) -> str:
    spoken = _spoken_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return spoken if spoken is not None else m.group(0)


def _speak_us_date(m: re.Match) -> str:
    month, day = int(m.group(1)), int(m.group(2))
    year_raw   = m.group(3)
    # 2-digit years: pivot 1969/2070 the same way Python's strptime("%y") does.
    if len(year_raw) == 2:
        yy = int(year_raw)
        year = 2000 + yy if yy < 70 else 1900 + yy
    else:
        year = int(year_raw)
    spoken = _spoken_date(year, month, day)
    return spoken if spoken is not None else m.group(0)


@dataclass
class DateDetector:
    """Recognizes ISO (2026-05-17) and US-slash (5/17/2026) date strings and
    speaks them as English. Out-of-range values pass through untouched.
    """
    name: str = "date"
    priority: int = 40

    def scan(self, text: str) -> list[Segment]:
        segs = _scan_regex(text, _DATE_ISO, _speak_iso_date, self.name)
        new: list[Segment] = []
        for s in segs:
            if isinstance(s, Claimed):
                new.append(s); continue
            new.extend(_scan_regex(s.raw, _DATE_US, _speak_us_date, self.name))
        return new


# ---------- phone numbers ----------

# US-style phones we recognize:
#   (555) 123-4567
#   555-123-4567   555.123.4567   555 123 4567
#   +1-800-555-0199   1-800-555-0199   +1 800 555 0199
#
# `_PHONE_CORE` captures the three groups of digits; `_PHONE_PREFIX`
# optionally matches a leading +1 or 1 with separator. We assemble both
# parts manually so we can reason about the country-code prefix without
# combinatorial regex pain.
_PHONE_PREFIX = re.compile(r"\+?1[-.\s]")
_PHONE = re.compile(
    r"""
    (?<!\d)                            # not glued to a longer digit run
    (?:\+?1[-.\s])?                    # optional country code "1" or "+1"
    \(?(\d{3})\)?[-.\s]                # area code
    (\d{3})[-.\s]                      # exchange
    (\d{4})                            # subscriber
    (?!\d)
    """,
    re.VERBOSE,
)

# Toll-free area codes spoken as a compact phrase rather than three digits.
# "800" reads as "eight hundred"; "888" as "eight eighty-eight"; etc.
_TOLL_FREE = {
    "800": "eight hundred",
    "888": "eight eighty-eight",
    "877": "eight seventy-seven",
    "866": "eight sixty-six",
    "855": "eight fifty-five",
    "844": "eight forty-four",
    "833": "eight thirty-three",
}

_DIGIT_NAMES = ("zero", "one", "two", "three", "four",
                "five", "six", "seven", "eight", "nine")


def _digits_spelled(s: str) -> str:
    return " ".join(_DIGIT_NAMES[int(d)] for d in s)


def _speak_phone(m: re.Match) -> str:
    area, exchange, subscriber = m.group(1), m.group(2), m.group(3)
    has_country = bool(_PHONE_PREFIX.match(m.group(0)))
    parts = []
    if has_country:
        parts.append("one")
    # Toll-free area codes: spoken as a compact phrase. Otherwise digit by digit.
    parts.append(_TOLL_FREE.get(area, _digits_spelled(area)))
    parts.append(_digits_spelled(exchange))
    parts.append(_digits_spelled(subscriber))
    return ", ".join(parts)


@dataclass
class PhoneDetector:
    """Recognizes US-style phone numbers and reads them digit-by-digit, with
    toll-free area codes (1-800, 1-888, ...) spoken as compact phrases.

    Runs after DateDetector but before NumberDetector so the embedded digits
    don't get currency/percent/unit treatment.
    """
    name: str = "phone"
    priority: int = 50

    def scan(self, text: str) -> list[Segment]:
        return _scan_regex(text, _PHONE, _speak_phone, self.name)


# ---------- numbers (currencies, percents, units) ----------

_DOLLAR    = re.compile(r"\$(\d[\d,]*\d|\d)(?:\.(\d+))?")
_PERCENT   = re.compile(r"(\d+(?:\.\d+)?)%")
_UNIT_GLUE = re.compile(r"(\d)([A-Za-zµ]+)\b")


def _spoken_dollar(m: re.Match) -> str:
    """Render a captured $amount as natural English.

    The interesting case is the decimal portion: in money, decimals are
    cents, not a fractional number. $1,000.50 is "one thousand dollars and
    fifty cents", not "one thousand point fifty dollars". We pad/truncate
    to two digits so $1,000.5 still reads as 50 cents.
    """
    whole, frac = m.group(1), m.group(2)
    if frac is None:
        return f"{whole} dollars"
    # Normalize to a 2-digit cents count: ".5" -> 50, ".50" -> 50, ".5000" -> 50.
    cents = int((frac + "00")[:2])
    whole_int = int(whole.replace(",", ""))
    if whole_int == 0 and cents > 0:
        return f"{cents} cents"
    if cents == 0:
        return f"{whole} dollars"
    return f"{whole} dollars and {cents} cents"


@dataclass
class NumberDetector:
    name: str = "number"
    priority: int = 60

    def scan(self, text: str) -> list[Segment]:
        segs = _scan_regex(text, _DOLLAR, _spoken_dollar, self.name)
        for pat, fmt in [
            (_PERCENT,   lambda m: f"{m.group(1)} percent"),
            (_UNIT_GLUE, lambda m: f"{m.group(1)} {m.group(2)}"),
        ]:
            new: list[Segment] = []
            for s in segs:
                if isinstance(s, Claimed):
                    new.append(s); continue
                new.extend(_scan_regex(s.raw, pat, fmt, self.name))
            segs = new
        return segs


# ---------- symbols ----------

_SYMBOL_TABLE = [
    (r"<->", " bidirectional "),
    (r"-->", " arrow "),
    (r"->",  " arrow "),
    (r"<=>", " if and only if "),
    (r"=>",  " implies "),
    (r"<=",  " less than or equal to "),
    (r">=",  " greater than or equal to "),
    (r"==",  " equals "),
    (r"!=",  " not equal to "),
    (r"&&",  " and "),
    (r"\|\|", " or "),
    (r"\.\.\.", " "),
]


@dataclass
class SymbolDetector:
    name: str = "symbol"
    priority: int = 70

    def scan(self, text: str) -> list[Segment]:
        new = text
        for pat, repl in _SYMBOL_TABLE:
            new = re.sub(pat, repl, new)
        return [Unclaimed(text)] if new == text else [Claimed(raw=text, spoken=new, detector=self.name)]
