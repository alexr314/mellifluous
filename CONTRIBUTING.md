# Contributing

Thanks for thinking about contributing. This is a small personal project so
the bar is low and informal.

## Scope

Things that fit:

- Bug fixes
- New detectors (citations, dates, acronyms, footnotes, anything else that
  reads badly out of the box)
- Better default pauses or policy presets
- Equation reader integrations for other providers (OpenAI, Anthropic, local)
- New TTS backends (subclass `mellifluous.Backend` and wire into
  `make_backend`; ElevenLabs, Cartesia, a PyTorch/CUDA local path, etc.)
- New domains: drop a file in `src/mellifluous/extras/domains/` with the
  field's acronyms, pronunciations, classifier regex hints, and an
  equation reader system prompt. See the existing files for the shape.
  Agents are very good at drafting these from a paper or two.
- Tests for things that have actually broken

Things that probably do not fit:

- Major rewrites. Open an issue first.

## Setup

```bash
git clone <repo>
cd mellifluous
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,llm]'
.venv/bin/pytest
```

The tests do not load any TTS model, so they run in seconds.

## Style

- No emoji, no em dashes, no smart quotes anywhere in source or docs.
- Add comments only when the *why* is non-obvious. Don't restate what the
  code does.
- Keep new dependencies optional unless they're genuinely required for the
  core path.

## New detectors

If you add a detector, also:

- Pick a `priority` that makes sense relative to the existing ones (see
  `mellifluous/detect/builtin.py`).
- Preserve text: `''.join(s.raw for s in scan(text)) == text`. The pipeline
  asserts this and will tell you immediately if you got it wrong.
- Add a smoke test under `tests/test_detect.py`.

## Pull requests

Keep them focused. One detector per PR, one bug fix per PR. A short
description of the user-visible change is enough; no template required.
