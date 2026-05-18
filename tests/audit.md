# Mellifluous Feature Audit

This document exercises every feature so we can hear them all in one pass.

## Paragraphs and line wrapping

This is the first paragraph. It is intentionally a bit long so we can hear
that there is a natural breath between paragraphs, rather than a single
continuous rush of words.

This is the second paragraph. The pause between these two should feel like
the kind of pause a human reader would naturally take.

## Lists

Unordered:

- the first item
- the second item, which is slightly longer to give a different rhythm
- the third item, the last one in this list

Ordered:

1. step one
2. step two
3. step three

## Block quote

> A good book is a good friend, but a great book is a great teacher.
> Reading should feel like being spoken to by a thoughtful narrator.

## Code

Here is a sentence containing inline code like `df.merge(left, right)` and a
short identifier `pandas.DataFrame`. After that, here is a fenced code block:

```python
def hello():
    print("world")
    return 42
```

## Horizontal rule

Below is a horizontal rule, which is rendered as a short silent pause.

---

After the rule we return to normal prose.

## Inline normalizations

Visit our docs at https://example.com/path?ref=audit for details, or email
alex@example.com. We had 25% growth this quarter, with revenue of $1,200.50
and a 10kg starter package. The condition `a == b && c != d` is true, and
control flows x -> y.

## Equations

The most famous equation in physics is $E = mc^2$. The standard normal
distribution has density \(\frac{1}{\sqrt{2\pi}} e^{-x^2/2}\). Block form:

$$ \int_a^b f'(x) \, dx = f(b) - f(a) $$

## Tables

A short table, read row by row:

| Quarter | Revenue | Growth |
|---------|---------|--------|
| Q1 | 1.2M | 8% |
| Q2 | 1.4M | 12% |
| Q3 | 1.5M | 7% |

The end.
