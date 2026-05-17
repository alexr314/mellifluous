# mellifluous

Markdown-to-speech for macOS Apple Silicon. Parses structure, normalizes
inline content, speaks with pauses that feel like real reading. TTS runs
locally on MLX with voice cloning.

Built by Alex Roman with help from Claude Opus 4.7.

## Requirements

- macOS on Apple Silicon (M1+)
- Python 3.10+
- ~2 GB free disk for the Qwen3-TTS weights (auto-downloaded on first run)

## Install

```bash
git clone <this repo>
cd mellifluous
python3 -m venv .venv
.venv/bin/pip install -e .
```

For LLM-backed equation reading (optional):

```bash
.venv/bin/pip install -e '.[llm]'
echo "GROQ_API_KEY=sk-..." > ~/.env
```

## Try it

A one-command demo that warms the model, asks for an optional Groq key, and
speaks a document exercising every feature:

```bash
python demo.py
```

## Use

```python
from mellifluous import Reader

r = Reader()
r.warm()                 # optional, loads the model now instead of on first speak
r.speak("# Hello\n\nThis is a *test*.")
```

`Reader()` picks the first voice in `voices/`. Pass `voice="alex"` to choose
explicitly. Other entry points:

```python
r.synthesize(text)       # iterator of AudioChunk (numpy float32 mono)
r.to_wav(text, "out.wav")
r.utterances(text)       # parsed, no audio
r.speak(token_iter, as_markdown=False)   # for LLM token streams
```

See `examples/` for runnable demos.

## Voices

A voice is a 5 to 30 second WAV clip the model clones from. Voices live in
`voices/<name>/sample.wav` (24 kHz mono PCM_16).

To add your own:

```bash
ffmpeg -i your_clip.mp3 -ar 24000 -ac 1 -c:a pcm_s16le voices/myname/sample.wav
```

Then `Reader(voice="myname")`. Override the voices directory with
`MELLIFLUOUS_VOICES_DIR=/path` or `Reader(voices_dir=Path("..."))`.

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

By default, equations are spoken as the word "equation". For real math, plug
an LLM into `EquationDetector`:

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

## Status and non-goals

Personal project, not on PyPI. macOS Apple Silicon only. The TTS code is from
[mlx-audio](https://github.com/Blaizzy/mlx-audio); the parser is
[markdown-it-py](https://github.com/executablebooks/markdown-it-py).

To port to Linux/CUDA, replace `mellifluous/synthesize/` with the equivalent
PyTorch path. The rest of the package has no Apple-specific code.
