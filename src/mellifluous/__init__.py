"""mellifluous: markdown to speech with pluggable TTS backends.

Quickstart:

    from mellifluous import Reader
    r = Reader()                                # OpenAI gpt-4o-mini-tts, voice "ash"
    r.speak("# Hello\\n\\nThis is a *test* with $E = mc^2$.")

    # Local Qwen3-TTS on Apple Silicon (install with `pip install mellifluous[local]`):
    r = Reader(engine="local", model="qwen-1.7b-8bit", voice="alex")
    r.warm()
    r.speak("...")

Pipeline:
  - mellifluous.parse       markdown -> Spans
  - mellifluous.detect      pluggable inline content recognition
  - mellifluous.vocalize    Spans -> Utterances (with structural pauses)
  - mellifluous.synthesize  Backend (OpenAI / MLX local / ...) -> AudioChunks
"""
from .reader     import Reader, list_voices, find_voice
from .utterance  import Utterance
from .vocalize   import Policy
from .detect     import (
    Pipeline, Detector, Segment, Claimed, Unclaimed,
    EquationDetector, UrlDetector, InlineCodeDetector,
    NumberDetector, SymbolDetector, DateDetector, PhoneDetector,
)
from .synthesize import (
    Backend, make_backend,
    Voice, Clone, Preset,
    AudioChunk,
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
    "DateDetector", "PhoneDetector",
    # Synthesize
    "Backend", "make_backend",
    "Voice", "Clone", "Preset",
    "AudioChunk",
    "play", "write_wav", "collect",
    "synthesize_utterances",
]
