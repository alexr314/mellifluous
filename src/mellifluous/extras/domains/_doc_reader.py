"""DocumentReader: one Agent per document, classifies + reads in one session.

The interesting trick is provider-side prompt caching: by reusing a single
Agent (and therefore a single conversation Context) across every equation
in a document, the leading prompt tokens (system + document context) are
identical across calls, so Groq / OpenAI / Anthropic all cache them. The
classifier call and every equation read in that document share the same
cached prefix.

Usage from the Reader integration layer:

    dr = DocumentReader(
        document_text=md,
        domains=load_domains(),
        llm_factory=lambda sys: Agent(llm=Groq(model="..."), system_prompt=sys),
    )
    domain_name = dr.classify()           # tier-1 regex, optional tier-2 LLM
    reader_fn = dr.read_equation         # bind as the EquationDetector reader
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Optional

from ._base import Domain, classify_tier1

log = logging.getLogger(__name__)


GENERIC_READER_PROMPT = """You convert LaTeX math expressions into a concise, natural English reading suitable for being spoken aloud by a text-to-speech system.

Rules:
- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.
- Use natural English. Avoid robotic literal readings.
- For fractions, say "X over Y" or "X divided by Y", whichever flows better.
- For sums, integrals, products: "sum from i equals 1 to n of ...".
- For powers: "x squared", "x cubed", "x to the n".
- For roots: "square root of X", "n-th root of X".
- Greek letters: spell them ("alpha", "beta", "lambda", ...).
- Common functions: "sine of x", "log of x", "natural log of x".
- For long expressions, group naturally with brief pauses (commas).
- Do not announce "the equation is..." or similar.
"""


_CLASSIFIER_INSTRUCTION = """You will help me read a technical document aloud. First, classify the document into one of these domains based on the content I'm about to share. Reply with ONLY the domain name as one of the choices below, or the word "generic" if none fits well.

Choices:
{choices}

Document excerpt:
---
{excerpt}
---

Reply with the single domain name."""


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _doc_excerpt(text: str, max_chars: int = 4000) -> str:
    """First chunk of the document for classifier context. Most papers
    declare their field in the first few paragraphs."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


class DocumentReader:
    """Per-document classifier + equation reader. Shares one Agent so
    provider prompt caching applies across calls in the same document.

    If `llm_factory` is None, classification is tier-1-only and equation
    reading falls back to the rule-based reader (no network needed).
    """

    def __init__(
        self,
        document_text: str,
        domains: dict[str, Domain],
        *,
        llm_factory: Optional[Callable[[str], "object"]] = None,
        cache_dir: Optional[Path] = Path.home() / ".cache" / "mellifluous" / "domain_equations",
        explicit_domain: Optional[str] = None,
    ):
        self.document_text = document_text
        self.domains = domains
        self.llm_factory = llm_factory
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._agent = None  # built lazily so doc with no equations costs nothing
        self._doc_hash = _hash(document_text)
        self._classified: Optional[str] = explicit_domain
        self._classification_done = explicit_domain is not None
        self._mem: dict[str, str] = {}

    # --- classification ---------------------------------------------------

    def classify(self) -> Optional[str]:
        """Return the chosen domain name (or None for generic).

        Tier 1: regex pattern hit count. Tier 2 (only if llm_factory is set
        AND tier 1 was ambiguous): ask the LLM to pick from the candidate
        list using a document excerpt.
        """
        if self._classification_done:
            return self._classified

        # Tier 1: pure regex
        winner = classify_tier1(self.document_text, self.domains)
        if winner is not None:
            log.info("domain classifier (tier 1): %s", winner)
            self._classified = winner
            self._classification_done = True
            return winner

        # Tier 2: LLM, if available
        if self.llm_factory is not None and self.domains:
            try:
                winner = self._classify_llm()
                if winner:
                    log.info("domain classifier (tier 2 LLM): %s", winner)
                    self._classified = winner
                    self._classification_done = True
                    return winner
            except Exception as e:
                log.warning("LLM classifier failed: %s", e)

        log.info("domain classifier: no winner, using generic")
        self._classification_done = True
        return None

    def _classify_llm(self) -> Optional[str]:
        choices = "\n".join(
            f"- {d.name}: {d.description}" for d in self.domains.values()
        )
        prompt = _CLASSIFIER_INSTRUCTION.format(
            choices=choices,
            excerpt=_doc_excerpt(self.document_text),
        )
        agent = self._ensure_agent()
        msg = agent.run(prompt)
        reply = (msg.text or "").strip().lower()
        for name in self.domains:
            if reply == name.lower():
                return name
        return None

    # --- equation reading -------------------------------------------------

    def read_equation(self, latex: str) -> str:
        """Return spoken English for a LaTeX equation, in the active domain.

        If no LLM is configured, falls through to the rule-based reader.
        """
        latex = latex.strip()
        if not latex:
            return ""

        if self.llm_factory is None:
            from mellifluous.detect.rule_reader import rule_based_reader
            return rule_based_reader(latex)

        # Cache by (active domain, latex). Same equation in different domains
        # reads differently, so domain has to be part of the key.
        domain_name = self.classify() or "generic"
        if latex in self._mem:
            return self._mem[latex]
        cp = self._cache_path(domain_name, latex)
        if cp is not None and cp.exists():
            try:
                data = json.loads(cp.read_text())
                if data.get("latex") == latex and data.get("domain") == domain_name:
                    spoken = data["spoken"]
                    self._mem[latex] = spoken
                    return spoken
            except Exception:
                pass

        try:
            agent = self._ensure_agent()
            msg = agent.run(f"Read this equation aloud: {latex}")
            spoken = (msg.text or "").strip() or "equation"
        except Exception as e:
            log.warning("equation reader LLM failed (%s); using rule-based", e)
            from mellifluous.detect.rule_reader import rule_based_reader
            spoken = rule_based_reader(latex)

        self._mem[latex] = spoken
        if cp is not None:
            try:
                cp.write_text(json.dumps({
                    "latex": latex, "spoken": spoken, "domain": domain_name,
                }))
            except Exception:
                pass
        return spoken

    # --- internals --------------------------------------------------------

    def _ensure_agent(self):
        """Build the Agent on first use with a system prompt that reflects
        the active domain. The Agent is reused for every subsequent call,
        so the system prompt + accumulating Context get cached by the
        provider."""
        if self._agent is not None:
            return self._agent
        domain_name = self._classified
        domain = self.domains.get(domain_name) if domain_name else None
        if domain is not None and domain.equation_reader_prompt:
            sys_prompt = domain.equation_reader_prompt
        else:
            sys_prompt = GENERIC_READER_PROMPT
        assert self.llm_factory is not None
        self._agent = self.llm_factory(sys_prompt)
        return self._agent

    def _cache_path(self, domain_name: str, latex: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{domain_name}_{_hash(latex)}.json"


# --- convenience: build a Groq-backed llm_factory ------------------------

def groq_llm_factory(model: str = "openai/gpt-oss-120b", api_key: Optional[str] = None):
    """Return an llm_factory(sys_prompt) -> Agent suitable for DocumentReader.

    Raises RuntimeError if orchestral-ai isn't installed or GROQ_API_KEY
    isn't set. Callers should catch that and fall back to llm_factory=None.
    """
    try:
        from orchestral import Agent
        from orchestral.llm import Groq
    except ImportError as e:
        raise RuntimeError(
            "orchestral-ai not installed. pip install 'mellifluous[llm]'"
        ) from e
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "no GROQ_API_KEY in env; set it or pass api_key= to groq_llm_factory"
        )

    def factory(system_prompt: str):
        return Agent(llm=Groq(model=model, api_key=key), system_prompt=system_prompt)
    return factory
