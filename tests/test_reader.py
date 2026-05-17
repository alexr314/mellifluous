"""Smoke tests for the Reader entry point (no audio, no model load)."""
from pathlib import Path
import pytest

from mellifluous import Reader, Preset, find_voice, list_voices


def test_lists_shipped_alex_voice():
    voices = list_voices()
    assert "alex" in voices


def test_finds_alex_voice_path():
    p = find_voice("alex")
    assert p.exists()
    assert p.name == "sample.wav"


def test_find_missing_voice_reports_available():
    with pytest.raises(FileNotFoundError) as e:
        find_voice("definitely-not-a-voice")
    assert "alex" in str(e.value)


def test_reader_picks_first_voice_without_loading_model():
    """Constructing a Reader configures the voice but should not touch MLX."""
    r = Reader()
    assert r.voice is not None
    # Sanity: it picked alex.
    assert "alex" in str(getattr(r.voice, "ref_audio", ""))


def test_reader_utterances_runs_without_audio_deps():
    """utterances() is pure markdown -> Utterance; no model load needed."""
    r = Reader()
    out = list(r.utterances("# Hi\n\nA paragraph."))
    roles = [u.role for u in out]
    assert "heading.1" in roles and "paragraph" in roles


def test_preset_model_without_voice_raises_with_helpful_message():
    with pytest.raises(TypeError, match="Available presets"):
        Reader(model="1.7b-custom-8bit")


def test_preset_voice_accepted_for_preset_model():
    r = Reader(model="1.7b-custom-8bit", voice=Preset("eric"))
    assert isinstance(r.voice, Preset)
    assert r.voice.speaker == "eric"


def test_custom_voices_dir_via_argument(tmp_path: Path):
    """Reader honors a passed voices_dir."""
    vdir = tmp_path / "voices"
    (vdir / "bob").mkdir(parents=True)
    (vdir / "bob" / "sample.wav").write_bytes(b"placeholder")
    r = Reader(voices_dir=vdir)
    assert "bob" in str(r.voice.ref_audio)


def test_env_var_overrides_voices_dir(tmp_path: Path, monkeypatch):
    vdir = tmp_path / "envvoices"
    (vdir / "carol").mkdir(parents=True)
    (vdir / "carol" / "sample.wav").write_bytes(b"placeholder")
    monkeypatch.setenv("MELLIFLUOUS_VOICES_DIR", str(vdir))
    assert list_voices() == ["carol"]
