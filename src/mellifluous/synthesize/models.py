"""Qwen3-TTS model registry."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


ModelKind = Literal["clone", "preset"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str            # HuggingFace repo (mlx-audio compatible)
    kind: ModelKind


MODELS: dict[str, ModelSpec] = {
    m.id: m for m in [
        ModelSpec("1.7b-8bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",        "clone"),
        ModelSpec("1.7b-6bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit",        "clone"),
        ModelSpec("1.7b-4bit",        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",        "clone"),
        ModelSpec("0.6b-8bit",        "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",        "clone"),
        ModelSpec("1.7b-custom-8bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "preset"),
    ]
}

DEFAULT_MODEL = "1.7b-8bit"

PRESET_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]
