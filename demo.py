#!/usr/bin/env python3
"""mellifluous demo: speak the audit document end-to-end.

Run from the repo root with the project's venv activated:

    python demo.py                          # default: OpenAI gpt-4o-mini-tts
    python demo.py --engine local           # macOS Apple Silicon: Qwen3-TTS via MLX
    python demo.py --voice nova             # any OpenAI preset
    python demo.py --instructions "calm, warm narrator"

The script checks its environment and tells you what to do if anything is
missing. On the first full run with --engine local it will:

  - prompt for a GROQ_API_KEY if you have one (optional)
  - load the TTS model (downloads ~2GB the first time)

With the default OpenAI backend it just needs OPENAI_API_KEY in the env.
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import textwrap
from pathlib import Path

try:
    from dotenv import load_dotenv
    for _p in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env", Path.home() / ".env"):
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass


REPO_ROOT      = Path(__file__).resolve().parent
AUDIT_PATH     = REPO_ROOT / "tests" / "audit.md"
RUN_AGAIN_HINT = "python demo.py"


def die(msg: str, *, code: int = 1) -> None:
    print(textwrap.dedent(msg).strip(), file=sys.stderr)
    sys.exit(code)


def check_venv() -> None:
    in_venv = (sys.prefix != getattr(sys, "base_prefix", sys.prefix))
    if not in_venv:
        die(f"""
            No Python virtual environment is active.

            From the repo root, set one up and re-run this script:

                python3 -m venv .venv
                # macOS/Linux:   source .venv/bin/activate
                # Windows:       .venv\\Scripts\\activate
                pip install -e .                  # OpenAI backend (default)
                pip install -e '.[local,llm]'     # add MLX local + Groq eq reader (macOS)
                {RUN_AGAIN_HINT}
        """)


def local_available() -> bool:
    """True iff the MLX local engine can run on this machine."""
    if sys.platform != "darwin":
        return False
    try:
        __import__("mlx_audio")
    except ImportError:
        return False
    return True


def check_install(engine: str) -> None:
    """Confirm the deps for the chosen engine import. If not, guide the user."""
    common = [
        ("mellifluous", "pip install -e ."),
        ("sounddevice", "pip install -e ."),
        ("soundfile",   "pip install -e ."),
        ("markdown_it", "pip install -e ."),
    ]
    if engine == "openai":
        common.append(("openai", "pip install -e ."))
    elif engine == "local":
        if sys.platform != "darwin":
            die("""
                --engine local requires macOS Apple Silicon (the MLX TTS engine
                only runs there). Use the default --engine openai on other
                platforms, or see the README for porting notes.
            """)
        common.append(("mlx_audio", "pip install -e '.[local]'"))

    missing = []
    for mod, hint in common:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, hint))
    if missing:
        names = ", ".join(m for m, _ in missing)
        hints = sorted({h for _, h in missing})
        die(f"""
            Missing Python packages: {names}.

            Install everything mellifluous needs, then re-run this script:

                {chr(10).join('                ' + h for h in hints).strip()}
                {RUN_AGAIN_HINT}
        """)


def check_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        die("""
            OPENAI_API_KEY is not set.

            Export it (or put it in a .env you load before running):

                export OPENAI_API_KEY=sk-...

            Or pick another engine:

                python demo.py --engine local        # macOS only
        """)


def check_ffmpeg_optional() -> None:
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


def build_reader(args, groq_key: str | None):
    """Construct the Reader with or without the LLM equation reader."""
    from mellifluous import (
        Reader, Policy, Pipeline,
        EquationDetector, UrlDetector, InlineCodeDetector,
        NumberDetector, SymbolDetector, DateDetector, PhoneDetector,
    )

    detectors = [
        UrlDetector(),
        InlineCodeDetector(),
        DateDetector(),
        PhoneDetector(),
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

    reader_kwargs = {
        "engine":  args.engine,
        "policy":  policy,
    }
    if args.model:
        reader_kwargs["model"] = args.model
    if args.voice:
        reader_kwargs["voice"] = args.voice
    if args.instructions:
        reader_kwargs["instructions"] = args.instructions
    return Reader(**reader_kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["openai", "local"], default=None,
                   help="TTS engine (default: local if available, else openai)")
    p.add_argument("--model", default=None,
                   help="model id (default: per-engine — gpt-4o-mini-tts / qwen-1.7b-8bit)")
    p.add_argument("--voice", default=None,
                   help="voice name (openai: ash/nova/sage/..., local: name in voices/)")
    p.add_argument("--instructions", default=None,
                   help="OpenAI gpt-4o-mini-tts tone steering, e.g. \"calm, warm narrator\"")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_venv()
    if args.engine is None:
        args.engine = "local" if local_available() else "openai"
        print(f"defaulting to --engine {args.engine}.")
    check_install(args.engine)
    if args.engine == "openai":
        check_openai_key()
    check_ffmpeg_optional()

    if not AUDIT_PATH.exists():
        die(f"""
            Audit document missing: {AUDIT_PATH}
            Re-clone the repo, or pass your own markdown to the Reader directly.
        """)

    groq_key = prompt_groq_key()
    reader = build_reader(args, groq_key)

    print()
    if args.engine == "local":
        print("loading TTS model (this can take ~10s warm, a few minutes cold")
        print("if the weights are not yet downloaded)...")
        import time
        t = time.time()
        reader.warm()
        print(f"ready in {time.time() - t:.1f}s.")
        print()
    else:
        print(f"engine: openai / {reader.model} / voice={reader.voice}")
        if reader.backend.instructions:
            print(f"instructions: {reader.backend.instructions!r}")
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
