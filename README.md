# mellifluous

Markdown-to-speech with pluggable TTS backends. Parses structure, normalizes
inline content, speaks with pauses that feel like real reading. Ships with
two backends: Qwen3-TTS on MLX for fully-local voice cloning on Apple
Silicon (preferred when available), and OpenAI cloud TTS as a cross-platform
fallback. `Reader()` picks whichever will work in your environment.

Built by Alex Roman with help from Claude Opus 4.7.

## Requirements

- Python 3.10+
- For the local backend (preferred, picked by default on Mac): macOS on
  Apple Silicon (M1+) and ~2 GB of disk for the Qwen3-TTS weights
  (auto-downloaded on first run). Install with `pip install -e '.[local]'`.
- For the OpenAI backend (fallback elsewhere, or if you prefer cloud TTS):
  an `OPENAI_API_KEY`. Runs on macOS, Linux, and Windows.

## Install

```bash
git clone <this repo>
cd mellifluous
python3 -m venv .venv
.venv/bin/pip install -e '.[local]'         # macOS Apple Silicon: local Qwen3-TTS
# or
.venv/bin/pip install -e .                  # OpenAI cloud only (any platform)
```

For LLM-backed equation reading (optional):

```bash
.venv/bin/pip install -e '.[llm]'
echo "GROQ_API_KEY=sk-..." > ~/.env
```

## Try it

A one-command demo that asks for any optional keys and speaks a document
exercising every feature:

```bash
python demo.py                              # auto: local on Mac, openai elsewhere
python demo.py --engine local               # macOS Apple Silicon: Qwen3-TTS via MLX
python demo.py --engine openai              # OpenAI cloud (needs OPENAI_API_KEY)
python demo.py --voice nova                 # OpenAI preset
python demo.py --instructions "calm, warm narrator"
```

## Use

```python
from mellifluous import Reader

r = Reader()                                          # local on Mac (Qwen3-TTS, voice cloned from voices/alex/), openai elsewhere
r.warm()                                              # local: load model now; openai: no-op
r.speak("# Hello\n\nThis is a *test*.")

# Force a backend:
r = Reader(engine="openai", voice="nova")             # OpenAI gpt-4o-mini-tts
r = Reader(engine="local", voice="alex")              # Qwen3-TTS (macOS Apple Silicon)
```

With the OpenAI backend, `voice` is a preset name (`"ash"`, `"nova"`, `"sage"`,
`"alloy"`, `"coral"`, `"echo"`, `"fable"`, `"onyx"`, `"shimmer"`, `"ballad"`,
`"verse"`). On `gpt-4o-mini-tts` you can also pass `instructions=` to steer
tone (`"calm, warm narrator"`, `"urgent, attention-grabbing"`, etc.), either
on the `Reader` or per-call to `speak()` / `synthesize()` / `to_wav()`.

With the local backend, `Reader(engine="local")` picks the first voice in `voices/`. Pass `voice="alex"` to choose
explicitly. Other entry points:

```python
r.synthesize(text)       # iterator of AudioChunk (numpy float32 mono)
r.to_wav(text, "out.wav")
r.utterances(text)       # parsed, no audio
r.speak(token_iter, as_markdown=False)   # for LLM token streams
```

See `examples/` for runnable demos.

## Voices

**OpenAI backend.** Voice is a preset name passed as a string. See the list
in the Use section above.

**Local backend.** A voice is a 5 to 30 second WAV clip the model clones
from. Voices live in `voices/<name>/sample.wav` (24 kHz mono PCM_16). To add
your own:

```bash
ffmpeg -i your_clip.mp3 -ar 24000 -ac 1 -c:a pcm_s16le voices/myname/sample.wav
```

Then `Reader(engine="local", voice="myname")`. Override the voices directory
with `MELLIFLUOUS_VOICES_DIR=/path` or `Reader(engine="local",
voices_dir=Path("..."))`.

The repo ships the `alex` voice (the author's own). Only clone voices you
have permission to use.

## Pacing

The `Policy` object controls per-element pause durations and verbosity. Tweak
per-call:

```python
from mellifluous import Reader, Policy
Reader(policy=Policy(paragraph_post=900, heading_pre={1: 1500})).speak(text)
```

## Detectors

Inline content (URLs, equations, inline code, currencies, ASCII operators) is
handled by a chain of `Detector` objects that run in priority order. Each
detector scans an unclaimed text region and returns claimed and unclaimed
segments; later detectors only see what's still unclaimed, so structural
detectors protect their content from substitutions.

Built-in:

| detector | what it does |
|---|---|
| `EquationDetector` | `$...$`, `$$...$$`, `\(...\)`, `\[...\]` get a pluggable reader |
| `UrlDetector` | URLs and emails become "link to example dot com" |
| `InlineCodeDetector` | `` `df.merge()` `` becomes "df merge" |
| `NumberDetector` | `$1,200`, `5%`, `10kg` |
| `SymbolDetector` | `->`, `==`, `&&`, etc. |

Add your own by implementing the `Detector` protocol and putting it in a
`Pipeline`. See `mellifluous/detect/builtin.py` for examples.

## LLM equation reader

By default, equations are read by a built-in rule-based reader that handles
common shapes (fractions, sums, integrals, square roots, Greek letters,
named operators, sub/superscripts) and falls back to the word "equation" for
anything more complex. For better math on arbitrary LaTeX, plug an LLM into
`EquationDetector`:

```python
from mellifluous import Reader, Policy, Pipeline, EquationDetector, \
    UrlDetector, InlineCodeDetector, NumberDetector, SymbolDetector
from mellifluous.extras.groq_equation_reader import make_reader

eq_reader = make_reader(model="openai/gpt-oss-120b")
policy = Policy(detectors=Pipeline([
    EquationDetector(reader=eq_reader),
    UrlDetector(), InlineCodeDetector(),
    NumberDetector(), SymbolDetector(),
]))
Reader(policy=policy).speak(r"The sum is $\sum_{i=1}^n i = \frac{n(n+1)}{2}$.")
```

Requires `pip install -e '.[llm]'` and `GROQ_API_KEY`. The example uses Groq's
`openai/gpt-oss-120b`, which returns in around a second and is cheap enough
at this volume to ignore. Results are cached on disk by LaTeX hash.

The reader is just a `Callable[[str], str]`, so any provider works. Write
your own one-line wrapper.

## Domains

A *domain* is a field of study that mellifluous knows how to read more
fluently than generic English. Each domain bundles:

- **Acronym table** — `POVM` → "positive operator valued measure",
  `2-SRE` → "two stabilizer Renyi entropy", etc.
- **Pronunciation overrides** — `qubit` → "kew bit",
  `Schrodinger` → "shro din ger".
- **Equation reader prompt** — the system prompt the LLM uses to read math
  in this field, in the style a domain expert would explain it at a
  whiteboard. For example, the high-energy theory domain summarizes
  Lagrangians by structural piece ("kinetic term, mass term, interaction
  vertex") rather than reading every index.
- **Classifier hints** — regex patterns that, if present in the document,
  suggest this field.

Use a domain explicitly:

```python
from mellifluous import Reader
r = Reader(domain="quantum_information")
r.speak("A POVM measures the qubit. The state is $\\ket{\\psi}$.")
```

Or auto-detect from the document:

```python
r = Reader(domain="auto")
r.speak(my_arxiv_paper_in_markdown)
```

Built-in domains: `quantum_information`, `high_energy_theory`,
`quantum_mechanics`, `bayesian_probability`, `linear_algebra`,
`exoplanet_astrophysics`. The classifier picks one per document; ambiguous
cases fall through to generic reading (no acronym substitution, generic
equation prompt).

When the `[llm]` extra is installed and `GROQ_API_KEY` is set, the chosen
domain's equation reader prompt is used; otherwise the rule-based reader
applies. Either way, the acronym and pronunciation substitutions run with
no LLM needed.

### Adding your own domain

Drop a Python file in `src/mellifluous/extras/domains/`:

```python
# src/mellifluous/extras/domains/condensed_matter.py
from mellifluous.extras.domains import Domain

DOMAIN = Domain(
    name="condensed_matter",
    description="Condensed matter physics: bands, phonons, magnons, ...",
    latex_patterns = (r"\\hat{H}", r"k_F", r"E_F", r"\\omega_q"),
    keyword_patterns = (r"\bFermi (?:surface|level)\b", r"\bband structure\b"),
    acronyms = {"DOS": "density of states", "DFT": "density functional theory"},
    pronunciations = {"phonon": "fo non"},
    equation_reader_prompt = "You convert LaTeX from condensed matter physics ...",
)
```

It is auto-discovered next time `Reader(domain="...")` is called. PRs welcome.

## Backends

The synthesize layer is a thin `Backend` ABC; the parse / detect / vocalize
pipeline is identical across backends. Two ship in the box:

| engine | model(s) | platform | notes |
|---|---|---|---|
| `"local"` (default on Mac) | Qwen3-TTS variants (1.7B / 0.6B, 4/6/8-bit, Base or CustomVoice) | macOS Apple Silicon | runs offline; voice cloning from a short WAV. |
| `"openai"` (default elsewhere) | `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd` | any | cloud, needs `OPENAI_API_KEY`. `gpt-4o-mini-tts` supports `instructions=` for tone steering. |

To add a new backend, subclass `mellifluous.Backend` (see
`src/mellifluous/synthesize/base.py`), wire it into `make_backend`, and the
rest of the pipeline picks it up unchanged.

## Status and non-goals

Personal project, not on PyPI. The parser is
[markdown-it-py](https://github.com/executablebooks/markdown-it-py); the
local backend wraps [mlx-audio](https://github.com/Blaizzy/mlx-audio); the
OpenAI backend wraps the [OpenAI Python SDK](https://github.com/openai/openai-python).

A Linux/CUDA local backend would mean a third `Backend` implementation
(PyTorch equivalent of the MLX one). PRs welcome; open an issue first.
