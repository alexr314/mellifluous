"""Reader: the high-level entry point.

    from mellifluous import Reader
    r = Reader()                                # defaults to OpenAI gpt-4o-mini-tts, voice "ash"
    r.speak("# Hello\\n\\nThis is a *test*.")

    # Use the local Qwen3-TTS backend (macOS Apple Silicon only):
    r = Reader(engine="local", model="qwen-1.7b-8bit", voice="alex")
    r.warm()
    r.speak("...")

    # Override per-call:
    r.speak("Heads up — server alert!", instructions="urgent, attention-grabbing")
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing  import Any, Iterable, Iterator, Optional, Union

from .parse      import parse_markdown
from .vocalize   import vocalize, Policy
from .utterance  import Utterance
from .synthesize import (
    AudioChunk, Backend, Voice, make_backend,
    synthesize_utterances, play, write_wav,
)


# Engine-specific defaults for `model`. When the caller doesn't pass model=
# explicitly, we look up by engine here.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini-tts",
    "local":  "qwen-1.7b-8bit",
}


_DOMAIN_CACHE: Optional[dict] = None  # populated on first use, never invalidated


def _all_domains() -> dict:
    """Auto-discover and cache the domain library. Importing lazily so the
    Reader stays fast for users who don't use domains."""
    global _DOMAIN_CACHE
    if _DOMAIN_CACHE is None:
        from .extras.domains import load_domains
        _DOMAIN_CACHE = load_domains()
    return _DOMAIN_CACHE


def _try_groq_factory():
    """Return a Groq llm_factory if the [llm] extra is installed and
    GROQ_API_KEY is set; otherwise None. Used to enable tier-2
    classification and LLM-backed equation reading."""
    try:
        from .extras.domains import groq_llm_factory
        return groq_llm_factory()
    except RuntimeError:
        return None  # not installed or no API key — fall back gracefully


def _build_domain_policy(base: Policy, domain_arg: str, markdown_text: str) -> Policy:
    """Per-document policy: base settings + a domain-aware detector pipeline.

    `domain_arg` is "auto" (classify the document) or a specific domain name.
    Unknown names raise ValueError. If "auto" classification finds no winner,
    we silently fall through to the base policy (no acronyms, generic
    equation reading) so an unclassifiable document still works.
    """
    from .detect import Pipeline, EquationDetector, AcronymDetector
    from .extras.domains import DocumentReader

    domains = _all_domains()
    if domain_arg != "auto":
        if domain_arg not in domains:
            raise ValueError(
                f"unknown domain {domain_arg!r}. "
                f"available: {sorted(domains)!r}, or 'auto' to classify."
            )

    llm_factory = _try_groq_factory()
    doc_reader = DocumentReader(
        document_text=markdown_text,
        domains=domains,
        llm_factory=llm_factory,
        explicit_domain=None if domain_arg == "auto" else domain_arg,
    )
    chosen = doc_reader.classify()  # None means generic / no acronyms
    if chosen is None:
        return base

    domain = domains[chosen]

    # Replace the base pipeline's EquationDetector (if any) with one wired
    # through the DocumentReader; insert an AcronymDetector built from the
    # domain's tables. Keep every other detector unchanged.
    new_detectors = []
    saw_equation = False
    for det in base.detectors.detectors:
        if isinstance(det, EquationDetector):
            new_detectors.append(EquationDetector(
                reader=doc_reader.read_equation,
                priority=det.priority,
                prefix=det.prefix,
                suffix=det.suffix,
            ))
            saw_equation = True
        else:
            new_detectors.append(det)
    if not saw_equation:
        new_detectors.append(EquationDetector(reader=doc_reader.read_equation))
    if domain.acronyms or domain.pronunciations:
        new_detectors.append(AcronymDetector(
            acronyms=dict(domain.acronyms),
            pronunciations=dict(domain.pronunciations),
        ))

    # Same Policy, new pipeline. dataclasses.replace preserves all other
    # fields (pauses, verbosity, etc.) so user overrides survive.
    from dataclasses import replace
    return replace(base, detectors=Pipeline(new_detectors))


def _default_engine() -> str:
    """Pick a backend that will actually work in the current environment.

    Preference order:
      1. macOS Apple Silicon with mlx_audio installed -> "local" (offline,
         voice cloning, sounds better to the author's ear)
      2. otherwise -> "openai" (works anywhere, needs OPENAI_API_KEY; if
         the key is missing the OpenAIBackend raises a clear error)
    """
    if sys.platform == "darwin":
        try:
            import mlx_audio  # noqa: F401
            return "local"
        except ImportError:
            pass
    return "openai"


# --- voice directory helpers (local-MLX-specific, re-exported for convenience) ---

def list_voices(voices_dir: Optional[Path] = None) -> list[str]:
    """Return the names of voices found in the local voices/ directory.

    Voices are subdirectories containing `sample.wav`. Only meaningful for
    the `local` engine; OpenAI voices are presets, not files.
    """
    from .synthesize.backends.local_mlx import list_voices as _ls
    return _ls(voices_dir)


def find_voice(name: str, voices_dir: Optional[Path] = None) -> Path:
    """Return the path to `voices/<name>/sample.wav`. Raises if missing.

    Only meaningful for the `local` engine.
    """
    from .synthesize.backends.local_mlx import find_voice as _fv
    return _fv(name, voices_dir)


# --- Reader -----------------------------------------------------------------

class Reader:
    """High-level markdown -> speech.

    Args:
        engine:   "local" or "openai". If omitted, picks "local" on macOS
                  Apple Silicon when mlx-audio is installed; otherwise
                  "openai" (which then needs OPENAI_API_KEY).
        model:    backend-specific model id. Defaults: gpt-4o-mini-tts (openai)
                  or qwen-1.7b-8bit (local).
        voice:    OpenAI: a preset name like "ash", "nova", "sage", etc.
                  Local: a voice name from voices/<name>/, or a Voice instance.
                  None picks a sensible default per backend.
        instructions: OpenAI gpt-4o-mini-tts only — tone steering string,
                  e.g. "calm, conversational assistant". Set a Reader-level
                  default; override per-call via speak()/synthesize().
        policy:   Vocalization policy (pauses, verbosity, detectors).
        domain:   None (default), "auto", or a domain name like
                  "quantum_information". When set, mellifluous loads the
                  domain's acronym table and pronunciation overrides, and
                  routes equation reading through a domain-aware LLM if
                  the [llm] extra is installed and GROQ_API_KEY is set.
                  "auto" classifies each document using regex hints (and
                  the LLM as a tie-breaker, if available).
        voices_dir: Local engine only — override the voices/ directory
                  (or set MELLIFLUOUS_VOICES_DIR).
        api_key:  OpenAI only — defaults to OPENAI_API_KEY env var.
        base_url: OpenAI only — for proxies / Azure / compat APIs.

    Backend-specific kwargs (instructions, api_key, base_url, voices_dir,
    streaming_interval, temperature) are forwarded to the chosen backend.
    """

    def __init__(
        self,
        *,
        engine: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
        policy: Optional[Policy] = None,
        domain: Optional[str] = None,
        **backend_kwargs: Any,
    ):
        self.engine = engine if engine is not None else _default_engine()
        self.model = model if model is not None else _DEFAULT_MODELS.get(self.engine)
        if self.model is None:
            raise ValueError(
                f"no default model registered for engine {engine!r}; pass model=..."
            )
        self.policy = policy or Policy()
        # Domain: None disables; "auto" classifies per document; any other
        # value pins to that domain name. Validated lazily in
        # _synthesize_markdown so importing the domain library doesn't
        # block Reader construction.
        self.domain = domain
        self.backend: Backend = make_backend(
            engine=self.engine,
            model=self.model,
            voice=voice,
            instructions=instructions,
            **backend_kwargs,
        )
        # Expose the resolved voice on the Reader for tests / introspection.
        # Each backend stashes its own (a str for openai, a Voice for local).
        self.voice = getattr(self.backend, "voice", voice)

    # ---------- public ----------

    def warm(self) -> None:
        """Pay one-time setup costs now (model load for local; no-op for openai)."""
        self.backend.warm()

    def utterances(self, markdown_text: str) -> Iterator[Utterance]:
        """Parse markdown and yield Utterances. No audio."""
        policy = self._policy_for(markdown_text)
        return vocalize(parse_markdown(markdown_text), policy=policy)

    def synthesize(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Iterator[AudioChunk]:
        """Yield AudioChunks for the given text.

        If `as_markdown` is True (default), `text_or_iter` is treated as a
        markdown string and pause-aware Utterances are produced. If False,
        the text is sent directly to TTS (useful for already-clean strings
        or LLM token streams that don't carry markdown structure).

        `voice` and `instructions` override Reader defaults for this call.
        """
        if as_markdown:
            if isinstance(text_or_iter, str):
                # Whole-document path: utterances flow through the bridge,
                # which doesn't know about voice/instructions overrides.
                # Apply them by swapping the backend's defaults for the
                # duration of the call.
                return self._synthesize_markdown(
                    text_or_iter, voice=voice, instructions=instructions
                )
            # Streaming markdown path: accumulate tokens into markdown
            # blocks (paragraphs / lists / code blocks), parse + vocalize
            # each block as it completes, and speak it while the next is
            # still streaming in. Preserves all markdown features
            # (headers, emphasis, equations, code) but starts audio early.
            return self._synthesize_markdown_streaming(
                text_or_iter, voice=voice, instructions=instructions,
            )
        return self.backend.synthesize(
            text_or_iter, voice=voice, instructions=instructions
        )

    def speak(
        self,
        text_or_iter: Union[str, Iterable[str]],
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> None:
        """Synthesize and play on the default audio output. Blocks until done."""
        play(self.synthesize(
            text_or_iter, as_markdown=as_markdown,
            voice=voice, instructions=instructions,
        ))

    def to_wav(
        self,
        text_or_iter: Union[str, Iterable[str]],
        path: str | Path,
        *,
        as_markdown: bool = True,
        voice: Optional[Union[str, Voice]] = None,
        instructions: Optional[str] = None,
    ) -> Path:
        """Synthesize and write a WAV file. Returns the path."""
        return write_wav(path, self.synthesize(
            text_or_iter, as_markdown=as_markdown,
            voice=voice, instructions=instructions,
        ))

    # ---------- internal ----------

    def _policy_for(self, markdown_text: str) -> Policy:
        """Return a Policy for processing `markdown_text`.

        Without `domain`, returns the Reader's static policy unchanged.
        With `domain`, builds a per-document policy: same pause / verbosity
        settings, but the detector pipeline gains a domain-specific
        AcronymDetector and (when an LLM is available) the EquationDetector
        is rewired through a DocumentReader so all equations in this
        document share one Agent / one cached prompt prefix.
        """
        if self.domain is None:
            return self.policy
        return _build_domain_policy(self.policy, self.domain, markdown_text)

    def _synthesize_markdown(
        self,
        markdown_text: str,
        *,
        voice: Optional[Union[str, Voice]],
        instructions: Optional[str],
    ) -> Iterator[AudioChunk]:
        # The bridge calls backend.synthesize(utt.text) without per-call
        # overrides. To honor per-call voice/instructions on the markdown
        # path, swap the backend's defaults for the duration of this call.
        backend = self.backend
        saved_voice = getattr(backend, "voice", None)
        saved_instr = getattr(backend, "instructions", None)
        if voice is not None and hasattr(backend, "voice"):
            backend.voice = backend._resolve_voice(voice) if hasattr(backend, "_resolve_voice") else voice
        if instructions is not None and hasattr(backend, "instructions"):
            backend.instructions = instructions
        try:
            yield from synthesize_utterances(backend, self.utterances(markdown_text))
        finally:
            if hasattr(backend, "voice"):
                backend.voice = saved_voice
            if hasattr(backend, "instructions"):
                backend.instructions = saved_instr

    def _synthesize_markdown_streaming(
        self,
        token_iter: Iterable[str],
        *,
        voice: Optional[Union[str, Voice]],
        instructions: Optional[str],
    ) -> Iterator[AudioChunk]:
        """Accumulate streamed tokens into markdown blocks, parse + speak
        each block as it completes. Audio starts playing well before the
        LLM finishes generating.

        Block boundary: a blank line outside any fenced code block. Lists,
        tables, blockquotes, headers, and paragraphs all end at a blank
        line, so block-by-block streaming preserves every markdown
        feature; only the document-level structure (which the parser uses
        to set inter-block pauses) is fragmented, and the per-block pauses
        produced by vocalize() still apply.

        Per-document concerns (domain auto-classification) currently fall
        back to per-block classification, since each block is parsed
        independently. For most LLM chat replies the dominant domain in
        each block is the same, so the chosen domain is stable.
        """
        from .synthesize._text import drain_markdown_blocks

        backend = self.backend
        saved_voice = getattr(backend, "voice", None)
        saved_instr = getattr(backend, "instructions", None)
        if voice is not None and hasattr(backend, "voice"):
            backend.voice = backend._resolve_voice(voice) if hasattr(backend, "_resolve_voice") else voice
        if instructions is not None and hasattr(backend, "instructions"):
            backend.instructions = instructions
        try:
            buf = ""
            for piece in token_iter:
                if not piece:
                    continue
                buf += piece
                blocks, buf = drain_markdown_blocks(buf)
                for block in blocks:
                    yield from synthesize_utterances(
                        backend, self.utterances(block),
                    )
            tail = buf.strip()
            if tail:
                # Final flush: whatever's left, even without a trailing
                # blank line, is a complete block at end-of-stream.
                yield from synthesize_utterances(
                    backend, self.utterances(tail),
                )
        finally:
            if hasattr(backend, "voice"):
                backend.voice = saved_voice
            if hasattr(backend, "instructions"):
                backend.instructions = saved_instr
