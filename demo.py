#!/usr/bin/env python3
"""mellifluous demo: speak the audit document end-to-end.

Run from the repo root with the project's venv activated:

    python demo.py

The script checks its environment and tells you what to do if anything is
missing. On the first full run it will:

  - prompt for a GROQ_API_KEY if you have one (optional)
  - load the TTS model (downloads ~2GB the first time)
  - speak the audit document with all features exercised
"""
from __future__ import annotations
import os
import sys
import shutil
import textwrap
from pathlib import Path


REPO_ROOT      = Path(__file__).resolve().parent
AUDIT_PATH     = REPO_ROOT / "tests" / "audit.md"
INSTALL_HINT   = "pip install -e '.[llm]'"
RUN_AGAIN_HINT = "python demo.py"


def die(msg: str, *, code: int = 1) -> None:
    print(textwrap.dedent(msg).strip(), file=sys.stderr)
    sys.exit(code)


def check_platform() -> None:
    if sys.platform != "darwin":
        die("""
            mellifluous targets macOS Apple Silicon. The TTS engine depends on
            MLX which only runs there. See README for porting notes.
        """)


def check_venv() -> None:
    in_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix))
    if not in_venv:
        die(f"""
            No Python virtual environment is active.

            From the repo root, set one up and re-run this script:

                python3 -m venv .venv
                source .venv/bin/activate
                {INSTALL_HINT}
                {RUN_AGAIN_HINT}
        """)


def check_install() -> None:
    """Confirm mellifluous + its core deps import. If not, guide the user."""
    missing = []
    for mod, hint in [
        ("mellifluous",  INSTALL_HINT),
        ("mlx_audio",    INSTALL_HINT),
        ("sounddevice",  INSTALL_HINT),
        ("soundfile",    INSTALL_HINT),
        ("markdown_it",  INSTALL_HINT),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, hint))
    if missing:
        names = ", ".join(m for m, _ in missing)
        die(f"""
            Missing Python packages: {names}.

            Install everything mellifluous needs, then re-run this script:

                {INSTALL_HINT}
                {RUN_AGAIN_HINT}
        """)


def check_ffmpeg_optional() -> None:
    """ffmpeg is not strictly required for the demo, but warn if absent
    since it shows up everywhere else in the project (voice conversion)."""
    if shutil.which("ffmpeg") is None:
        print("(note: ffmpeg not on PATH. The demo itself doesn't need it,")
        print(" but the README's voice-cloning steps do.)")


def prompt_groq_key() -> str | None:
    """Returns a Groq API key or None. Tries env first, then prompts."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key

    print()
    print("Optional: a Groq API key gives the highest-quality equation reading,")
    print("rendering anything from simple identities to a Feynman path integral")
    print("as natural English. The free tier is plenty.")
    print()
    print("Without a key, we fall back to a built-in rule-based reader that")
    print("handles common shapes (fractions, sums, roots, Greek letters, etc.)")
    print("and says 'equation' for anything more complex.")
    print()
    print("Get a key at https://console.groq.com/keys (or skip with empty input).")
    print()
    try:
        entered = input("GROQ_API_KEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not entered:
        return None
    os.environ["GROQ_API_KEY"] = entered
    return entered


def build_reader(groq_key: str | None):
    """Construct the Reader with or without the LLM equation reader."""
    from mellifluous import (
        Reader, Policy, Pipeline,
        EquationDetector, UrlDetector, InlineCodeDetector,
        NumberDetector, SymbolDetector,
    )

    detectors = [
        UrlDetector(),
        InlineCodeDetector(),
        NumberDetector(),
        SymbolDetector(),
    ]
    if groq_key:
        try:
            from mellifluous.extras.groq_equation_reader import make_reader
            eq_reader = make_reader(model="openai/gpt-oss-120b", api_key=groq_key)
            detectors.insert(0, EquationDetector(reader=eq_reader))
            print("equation reader: Groq (openai/gpt-oss-120b)")
        except Exception as e:
            print(f"(could not build Groq equation reader: {e})")
            print("(falling back to the built-in rule-based reader)")
            detectors.insert(0, EquationDetector())
    else:
        detectors.insert(0, EquationDetector())
        print("equation reader: built-in rule-based fallback")

    policy = Policy(
        table_max_rows_to_read=10,
        detectors=Pipeline(detectors),
    )
    return Reader(policy=policy)


def main() -> None:
    check_platform()
    check_venv()
    check_install()
    check_ffmpeg_optional()

    if not AUDIT_PATH.exists():
        die(f"""
            Audit document missing: {AUDIT_PATH}
            Re-clone the repo, or pass your own markdown to the Reader directly.
        """)

    groq_key = prompt_groq_key()
    reader = build_reader(groq_key)

    print()
    print("loading TTS model (this can take ~10s warm, a few minutes cold")
    print("if the weights are not yet downloaded)...")
    import time
    t = time.time()
    reader.warm()
    print(f"ready in {time.time() - t:.1f}s.")
    print()

    md = AUDIT_PATH.read_text()
    print("speaking the audit document. Listen for:")
    print("  - paragraph and heading pauses")
    print("  - list item rhythm")
    print("  - blockquote brackets")
    print("  - code block summary")
    print("  - inline URLs, currency, percentages, operators")
    print("  - equations (natural reading if Groq is configured)")
    print("  - tables (announced row by row)")
    print()

    reader.speak(md)
    print()
    print("done.")


if __name__ == "__main__":
    main()
