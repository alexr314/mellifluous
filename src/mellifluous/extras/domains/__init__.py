"""Domain library: field-specific knowledge mellifluous can use to read
text more fluently.

A Domain bundles acronym tables, pronunciation overrides, equation-reader
prompts, and regex hints used to classify a document into the field. Domain
files in this package are auto-discovered.

    from mellifluous.extras.domains import load_domains, classify_tier1, Domain

    domains = load_domains()           # all built-in domains
    name = classify_tier1(text, domains)  # tier-1 classifier (regex only)

To add your own field, drop a file in this directory that defines a
module-level `DOMAIN = Domain(...)`. See any of the built-ins (e.g.
quantum_information.py) for a template.
"""
from ._base import Domain, load_domains, classify_tier1
from ._doc_reader import DocumentReader, groq_llm_factory, GENERIC_READER_PROMPT

__all__ = [
    "Domain", "load_domains", "classify_tier1",
    "DocumentReader", "groq_llm_factory", "GENERIC_READER_PROMPT",
]
