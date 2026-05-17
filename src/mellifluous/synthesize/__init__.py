"""Synthesize layer: pluggable TTS backends (OpenAI cloud, MLX local, ...)
streaming AudioChunks to a sink (play / write_wav / collect).
"""
from .types     import AudioChunk
from .voices    import Voice, Clone, Preset
from .base      import Backend, make_backend
from .sinks     import play, write_wav, collect
from .bridge    import synthesize_utterances

__all__ = [
    "AudioChunk",
    "Voice", "Clone", "Preset",
    "Backend", "make_backend",
    "play", "write_wav", "collect",
    "synthesize_utterances",
]
