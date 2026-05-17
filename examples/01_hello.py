"""Simplest possible example: read a string of markdown aloud."""
from mellifluous import Reader

r = Reader()                  # picks the first voice in voices/
r.warm()                       # optional: pay the model load up front
r.speak("""
# Hello!

This is **mellifluous**, a markdown-to-speech reader for macOS Apple Silicon.

It handles structural pauses, lists, blockquotes, and inline code without
running everything together into an unbroken wall of words.

> The most famous equation in physics is `E = mc^2`. We'll do better with math
> in the smart example.
""")
