"""mellifluous: markdown to speech for macOS Apple Silicon.

Quickstart:

    from mellifluous import Reader
    r = Reader()                           # picks the first voice in voices/
    r.warm()                               # optional: pay the model load up front
    r.speak("# Hello\\n\\nThis is a *test* with $E = mc^2$.")

The Reader class is the one-stop entry point. For advanced control, the
submodules are also exposed:
  - mellifluous.parse       markdown -> Spans
  - mellifluous.detect      pluggable inline content recognition
  - mellifluous.vocalize    Spans -> Utterances (with structural pauses)
  - mellifluous.synthesize  Qwen3-TTS streaming + audio sinks
"""
from .reader     import Reader, list_voices, find_voice
from .utterance  import Utterance
from .vocalize   import Policy
from .detect     import (
    Pipeline, Detector, Segment, Claimed, Unclaimed,
    EquationDetector, UrlDetector, InlineCodeDetector,
    NumberDetector, SymbolDetector,
)
from .synthesize import (
    Clone, Preset, Voice,
    DEFAULT_MODEL, MODELS, PRESET_SPEAKERS,
    AudioChunk, Streamer,
    play, write_wav, collect,
    synthesize_utterances,
)
from .parse      import parse_markdown
from .vocalize   import vocalize


__all__ = [
    # Top-level entry point
    "Reader", "list_voices", "find_voice",
    # Pipeline pieces
    "parse_markdown", "vocalize",
    "Utterance", "Policy",
    # Detectors
    "Pipeline", "Detector", "Segment", "Claimed", "Unclaimed",
    "EquationDetector", "UrlDetector", "InlineCodeDetector",
    "NumberDetector", "SymbolDetector",
    # Synthesize
    "Clone", "Preset", "Voice",
    "DEFAULT_MODEL", "MODELS", "PRESET_SPEAKERS",
    "AudioChunk", "Streamer",
    "play", "write_wav", "collect",
    "synthesize_utterances",
]
