"""Smoke tests for the Reader entry point and backend selection.

Tests are split by backend so the openai-only tests run on any platform
(no MLX install required) while the local-MLX tests are gated to macOS.
"""
from pathlib import Path
import sys
import pytest

from mellifluous import (
    Reader, Preset, Clone, Voice,
    find_voice, list_voices,
    Backend, make_backend,
)


# --- shared fixtures --------------------------------------------------------

@pytest.fixture
def fake_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")


# --- openai backend ---------------------------------------------------------

class TestOpenAIBackend:
    def test_reader_default_is_openai(self, fake_openai_key):
        r = Reader()
        assert r.engine == "openai"
        assert r.model == "gpt-4o-mini-tts"
        assert r.voice == "ash"

    def test_reader_custom_openai_voice(self, fake_openai_key):
        r = Reader(voice="nova")
        assert r.voice == "nova"

    def test_reader_invalid_openai_voice_raises(self, fake_openai_key):
        with pytest.raises(ValueError, match="unknown openai voice"):
            Reader(voice="definitely-not-a-voice")

    def test_reader_invalid_openai_model_raises(self, fake_openai_key):
        with pytest.raises(ValueError, match="unknown model"):
            Reader(model="not-a-real-model")

    def test_reader_instructions_passed_through(self, fake_openai_key):
        r = Reader(instructions="calm, conversational")
        assert r.backend.instructions == "calm, conversational"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            Reader(engine="openai")

    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = Reader(api_key="sk-explicit")
        assert r.voice == "ash"

    def test_utterances_runs_without_audio_deps(self, fake_openai_key):
        """utterances() is pure markdown -> Utterance; no network needed."""
        r = Reader()
        out = list(r.utterances("# Hi\n\nA paragraph."))
        roles = [u.role for u in out]
        assert "heading.1" in roles and "paragraph" in roles


# --- factory ----------------------------------------------------------------

class TestMakeBackend:
    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="unknown engine"):
            make_backend(engine="nonsense", model="whatever")

    def test_openai_factory(self, fake_openai_key):
        b = make_backend(engine="openai", model="gpt-4o-mini-tts")
        assert isinstance(b, Backend)
        assert b.sample_rate == 24000


# --- local MLX backend (macOS only) ----------------------------------------

mlx_only = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="local engine requires mlx-audio (macOS Apple Silicon only)",
)


@mlx_only
class TestLocalMLXBackend:
    def test_lists_shipped_alex_voice(self):
        voices = list_voices()
        assert "alex" in voices

    def test_finds_alex_voice_path(self):
        p = find_voice("alex")
        assert p.exists()
        assert p.name == "sample.wav"

    def test_find_missing_voice_reports_available(self):
        with pytest.raises(FileNotFoundError) as e:
            find_voice("definitely-not-a-voice")
        assert "alex" in str(e.value)

    def test_reader_picks_first_voice_without_loading_model(self):
        """Constructing a local Reader configures the voice but should not touch MLX."""
        r = Reader(engine="local")
        assert r.voice is not None
        assert "alex" in str(getattr(r.voice, "ref_audio", ""))

    def test_reader_utterances_runs_without_audio_deps(self):
        r = Reader(engine="local")
        out = list(r.utterances("# Hi\n\nA paragraph."))
        roles = [u.role for u in out]
        assert "heading.1" in roles and "paragraph" in roles

    def test_preset_model_without_voice_raises_with_helpful_message(self):
        with pytest.raises(TypeError, match="Available presets"):
            Reader(engine="local", model="qwen-1.7b-custom-8bit")

    def test_preset_voice_accepted_for_preset_model(self):
        r = Reader(engine="local", model="qwen-1.7b-custom-8bit", voice=Preset("eric"))
        assert isinstance(r.voice, Preset)
        assert r.voice.speaker == "eric"

    def test_custom_voices_dir_via_argument(self, tmp_path: Path):
        vdir = tmp_path / "voices"
        (vdir / "bob").mkdir(parents=True)
        (vdir / "bob" / "sample.wav").write_bytes(b"placeholder")
        r = Reader(engine="local", voices_dir=vdir)
        assert "bob" in str(r.voice.ref_audio)

    def test_env_var_overrides_voices_dir(self, tmp_path: Path, monkeypatch):
        vdir = tmp_path / "envvoices"
        (vdir / "carol").mkdir(parents=True)
        (vdir / "carol" / "sample.wav").write_bytes(b"placeholder")
        monkeypatch.setenv("MELLIFLUOUS_VOICES_DIR", str(vdir))
        assert list_voices() == ["carol"]
