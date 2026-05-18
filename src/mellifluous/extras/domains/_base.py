"""The Domain object: everything mellifluous needs to know to sound fluent
in a particular field.

A Domain bundles four pieces of field-specific knowledge:

  1. Classification hints   - how to detect that a document is about this field
  2. Acronym table           - field-specific shorthand that English-only TTS mangles
  3. Pronunciation overrides - words the TTS reliably mispronounces
  4. Equation reader prompt  - how a domain expert would read an equation aloud

Domains are auto-discovered: drop a Python file in this package that defines
a module-level `DOMAIN = Domain(...)` and it shows up in load_domains().

A domain file is the right place to add other field-specific knowledge later
(citation styles, unit conventions, ...). Add fields to Domain and have the
relevant detector consume them; existing domain files keep working since the
new fields default to empty.
"""
from __future__ import annotations
import importlib
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Domain:
    """A field of study, with everything mellifluous needs to read it well."""

    name: str
    """Short kebab- or snake-case identifier, e.g. 'quantum_information'."""

    description: str
    """One-sentence human-readable description, also used as a classifier hint."""

    # --- classifier inputs ---
    latex_patterns: tuple[str, ...] = ()
    """Regex patterns that, if present in the document's LaTeX, suggest this
    field. Use raw strings: r'\\\\ket', r'\\\\partial_\\\\mu'."""

    keyword_patterns: tuple[str, ...] = ()
    """Regex patterns that, if present in the document's prose, suggest this
    field. Word-bounded usually: r'\\bdensity matrix\\b'."""

    # --- substitution tables ---
    acronyms: dict[str, str] = field(default_factory=dict)
    """Field-specific shorthand -> spoken form. Case-sensitive, word-bounded
    matching: 'POVM' matches but 'povm' doesn't. Plurals match: 'POVMs'
    reads as the singular form followed by an 's' sound."""

    pronunciations: dict[str, str] = field(default_factory=dict)
    """Word -> respelling. Case-insensitive, word-bounded. Use for words the
    TTS reliably mangles: {'qubit': 'kew bit', 'ansatz': 'ahn zats'}."""

    # --- equation reading ---
    equation_reader_prompt: Optional[str] = None
    """System prompt fragment appended to the equation reader's prompt when
    this domain is active. None means use the generic reader."""


# --- loader ---------------------------------------------------------------

def load_domains() -> dict[str, Domain]:
    """Discover and load every Domain in this package.

    Each domain file in mellifluous.extras.domains must export a module-level
    `DOMAIN = Domain(...)`. Modules starting with '_' (like this one) are
    skipped.
    """
    import mellifluous.extras.domains as pkg
    out: dict[str, Domain] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
        dom = getattr(mod, "DOMAIN", None)
        if isinstance(dom, Domain):
            out[dom.name] = dom
    return out


# --- classifier (tier-1: pure regex, no LLM) ------------------------------

def classify_tier1(text: str, domains: dict[str, Domain]) -> Optional[str]:
    """Return the winning domain name based purely on regex pattern hits.

    Counts how many of each domain's latex_patterns + keyword_patterns appear
    in `text`. Returns the domain with the most hits, or None if every domain
    scored zero or the top two tie. The caller can fall back to an LLM
    classifier in the ambiguous case.
    """
    scores: dict[str, int] = {}
    for name, dom in domains.items():
        n = 0
        for p in dom.latex_patterns:
            n += len(re.findall(p, text))
        for p in dom.keyword_patterns:
            n += len(re.findall(p, text, re.IGNORECASE))
        if n > 0:
            scores[name] = n
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]
