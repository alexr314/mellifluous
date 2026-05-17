"""Synthesize layer: Qwen3-TTS on MLX, streaming chunks, audio sinks."""
from .types     import AudioChunk
from .models    import MODELS, DEFAULT_MODEL, PRESET_SPEAKERS, ModelSpec
from .voices    import Voice, Clone, Preset
from .streamer  import Streamer
from .sinks     import play, write_wav, collect
from .bridge    import synthesize_utterances

__all__ = [
    "AudioChunk",
    "MODELS", "DEFAULT_MODEL", "PRESET_SPEAKERS", "ModelSpec",
    "Voice", "Clone", "Preset",
    "Streamer",
    "play", "write_wav", "collect",
    "synthesize_utterances",
]
