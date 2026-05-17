"""Warm-up + REPL: type text, hear it spoken with sub-second TTFA."""
import time
from mellifluous import Reader

r = Reader()
print("loading model...", flush=True)
t = time.time()
r.warm()
print(f"warm in {time.time()-t:.1f}s. type markdown, blank line or Ctrl-D to quit.\n", flush=True)

while True:
    try:
        text = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not text:
        break
    t0 = time.time()
    r.speak(text)
    print(f"  ({time.time()-t0:.2f}s)", flush=True)
