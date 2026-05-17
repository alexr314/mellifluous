"""A pluggable EquationDetector reader backed by Groq via orchestral-ai.

    from mellifluous import Reader, Policy, Pipeline, EquationDetector, \\
        UrlDetector, InlineCodeDetector, NumberDetector, SymbolDetector
    from mellifluous.extras.groq_equation_reader import make_reader

    eq_reader = make_reader(model="openai/gpt-oss-120b")
    detectors = Pipeline([
        EquationDetector(reader=eq_reader),
        UrlDetector(), InlineCodeDetector(),
        NumberDetector(), SymbolDetector(),
    ])
    r = Reader(policy=Policy(detectors=detectors))
    r.warm()
    r.speak(some_markdown_with_math)

Caching: reader memoizes by exact LaTeX string in-process, and persists on
disk under ~/.cache/mellifluous/equations/ unless cache_dir=None.

Provider key: GROQ_API_KEY env var (auto-loaded from common .env locations
if python-dotenv is installed). Or pass api_key= explicitly.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

# Best-effort .env loading.
try:
    from dotenv import load_dotenv
    for p in (Path.cwd() / ".env", Path.home() / ".env"):
        if p.exists():
            load_dotenv(p)
            break
except ImportError:
    pass


DEFAULT_SYSTEM_PROMPT = """You convert LaTeX math expressions into a concise, natural English reading suitable for being spoken aloud by a text-to-speech system.

Rules:
- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.
- Use natural English. Avoid robotic literal readings.
- For fractions, say "X over Y" or "X divided by Y", whichever flows better.
- For sums, integrals, products: "sum from i equals 1 to n of ...", etc.
- For powers: "x squared", "x cubed", "x to the n", "x to the n-th power".
- For roots: "square root of X", "n-th root of X".
- Greek letters: spell them ("alpha", "beta", "lambda", ...).
- Common functions: "sine of x", "log of x", "natural log of x".
- For long expressions, group naturally with brief pauses (commas).
- Do not announce "the equation is..." or similar.

Examples:
  Input:  E = mc^2
  Output: E equals m c squared.

  Input:  \\frac{1}{2}
  Output: one half.

  Input:  \\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}
  Output: The sum from i equals 1 to n of i squared equals n times n plus one times two n plus one, all divided by six.

  Input:  \\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
  Output: The integral from zero to infinity of e to the minus x squared d x equals the square root of pi over two."""


def make_reader(
    *,
    model: str = "openai/gpt-oss-120b",
    api_key: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    cache_dir: Optional[Path] = Path.home() / ".cache" / "mellifluous" / "equations",
    on_call: Optional[Callable[[str, str], None]] = None,
) -> Callable[[str], str]:
    """Build a (latex -> spoken English) reader. Returned function is callable.

    `cache_dir=None` disables on-disk caching (in-memory cache always on).
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
            "no Groq API key. Set GROQ_API_KEY in your env or pass api_key=..."
        )

    agent = Agent(llm=Groq(model=model, api_key=key), system_prompt=system_prompt)
    mem: dict[str, str] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(latex: str) -> Optional[Path]:
        if cache_dir is None: return None
        h = hashlib.sha256(latex.encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"{h}.json"

    def reader(latex: str) -> str:
        latex = latex.strip()
        if not latex: return ""
        if latex in mem: return mem[latex]
        cp = _cache_path(latex)
        if cp and cp.exists():
            try:
                data = json.loads(cp.read_text())
                if data.get("latex") == latex:
                    spoken = data["spoken"]
                    mem[latex] = spoken
                    return spoken
            except Exception:
                pass
        msg = agent.run(latex)
        spoken = (msg.text or "").strip()
        mem[latex] = spoken
        if cp:
            try:
                cp.write_text(json.dumps({"latex": latex, "spoken": spoken, "model": model}))
            except Exception:
                pass
        if on_call:
            try: on_call(latex, spoken)
            except Exception: pass
        return spoken

    return reader
