"""TTS backend implementations.

Each module here exports a `Backend` subclass. They are imported lazily by
`mellifluous.synthesize.base.make_backend` so the package itself imports
cleanly even when an engine's heavy/platform-specific dependencies (mlx,
openai SDK) are missing.
"""
