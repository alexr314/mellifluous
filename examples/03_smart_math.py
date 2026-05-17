"""Plug an LLM-backed equation reader (Groq via orchestral-ai) into the
EquationDetector so equations get spoken in natural language.

Requires: pip install 'mellifluous[llm]'  and  export GROQ_API_KEY=...
"""
import sys

from mellifluous import (
    Reader, Policy, Pipeline,
    EquationDetector, UrlDetector, InlineCodeDetector,
    NumberDetector, SymbolDetector,
)
from mellifluous.extras.groq_equation_reader import make_reader

eq_reader = make_reader(
    model="openai/gpt-oss-120b",
    on_call=lambda lx, sp: print(f"  eq: {lx!r}\n      -> {sp!r}", flush=True),
)
policy = Policy(detectors=Pipeline([
    EquationDetector(reader=eq_reader),
    UrlDetector(),
    InlineCodeDetector(),
    NumberDetector(),
    SymbolDetector(),
]))

md = sys.stdin.read() if not sys.stdin.isatty() else r"""# A short tour of equations

The most famous equation in physics is $E = mc^2$. It tells us that mass and
energy are two forms of the same thing.

## Calculus

Newton and Leibniz gave us the fundamental theorem:

$$ \int_a^b f'(x) \, dx = f(b) - f(a) $$

The standard normal distribution has density \(\frac{1}{\sqrt{2\pi}} e^{-x^2/2}\).
"""

r = Reader(policy=policy)
print("loading model...", flush=True)
r.warm()
r.speak(md)
