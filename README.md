# mellifluous

Markdown-to-speech with pluggable TTS backends. Parses structure, normalizes
inline content, speaks with pauses that feel like real reading. Ships with
two backends: OpenAI cloud TTS (default, runs anywhere) and Qwen3-TTS on MLX
for fully-local voice cloning on Apple Silicon.

Built by Alex Roman with help from Claude Opus 4.7.

## Requirements

- Python 3.10+
- For the default OpenAI backend: an `OPENAI_API_KEY`. Runs on macOS, Linux,
  and Windows.
- For the optional local backend: macOS on Apple Silicon (M1+) and ~2 GB of
  disk for the Qwen3-TTS weights (auto-downloaded on first run).

## Install

```bash
git clone <this repo>
cd mellifluous
python3 -m venv .venv
.venv/bin/pip install -e .                  # OpenAI backend (default)
```

For local TTS on Apple Silicon:

```bash
.venv/bin/pip install -e '.[local]'         # adds mlx-audio
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
export OPENAI_API_KEY=sk-...
python demo.py                              # OpenAI (default, cross-platform)
python demo.py --engine local               # macOS Apple Silicon: Qwen3-TTS via MLX
python demo.py --voice nova                 # any OpenAI preset
python demo.py --instructions "calm, warm narrator"
```

## Use

```python
from mellifluous import Reader

r = Reader()                                          # OpenAI gpt-4o-mini-tts, voice "ash"
r.speak("# Hello\n\nThis is a *test*.")

# Local Qwen3-TTS on Apple Silicon, voice cloned from voices/alex/sample.wav:
r = Reader(engine="local", voice="alex")
r.warm()                                              # loads the model now instead of on first speak
r.speak("# Hello\n\nThis is a *test*.")
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

## Backends

The synthesize layer is a thin `Backend` ABC; the parse / detect / vocalize
pipeline is identical across backends. Two ship in the box:

| engine | model(s) | platform | notes |
|---|---|---|---|
| `"openai"` (default) | `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd` | any | cloud, needs `OPENAI_API_KEY`. `gpt-4o-mini-tts` supports `instructions=` for tone steering. |
| `"local"` | Qwen3-TTS variants (1.7B / 0.6B, 4/6/8-bit, Base or CustomVoice) | macOS Apple Silicon | runs offline; voice cloning from a short WAV. |

To add a new backend, subclass `mellifluous.Backend` (see
`src/mellifluous/synthesize/base.py`), wire it into `make_backend`, and the
rest of the pipeline picks it up unchanged.

## Status and non-goals

Personal project, not on PyPI. The parser is
[markdown-it-py](https://github.com/executablebooks/markdown-it-py); the
local backend wraps [mlx-audio](https://github.com/Blaizzy/mlx-audio); the
default backend wraps the [OpenAI Python SDK](https://github.com/openai/openai-python).

A Linux/CUDA local backend would mean a third `Backend` implementation
(PyTorch equivalent of the MLX one). PRs welcome; open an issue first.
