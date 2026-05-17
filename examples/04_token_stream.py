"""Feed a simulated LLM token stream. Speech starts as soon as the first
sentence is complete, not when the full text is in.
"""
import time
from mellifluous import Reader

def fake_llm_tokens():
    text = (
        "Good morning. "
        "Here's a quick summary of your day. "
        "You have three meetings, "
        "one urgent email, "
        "and a reminder to call the bank."
    )
    for i in range(0, len(text), 2):
        yield text[i:i + 2]
        time.sleep(0.05)

r = Reader()
r.warm()
# as_markdown=False because the LLM is producing plain prose, not markdown.
r.speak(fake_llm_tokens(), as_markdown=False)
