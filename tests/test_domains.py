"""Tests for the domain library: loading, classification, acronym
substitution, and Reader integration.

No tests in this file hit the network. The LLM classifier and equation
reader paths are exercised separately; here we cover only what works
offline (tier-1 regex classification + the substitution detectors).
"""
from __future__ import annotations
import pytest

from mellifluous.detect import (
    Pipeline, AcronymDetector, EquationDetector, normalize_text,
)
from mellifluous.extras.domains import (
    Domain, load_domains, classify_tier1, DocumentReader,
)


# --- Domain object + loader -------------------------------------------------

class TestDomainLoader:
    def test_all_builtin_domains_load(self):
        ds = load_domains()
        assert "quantum_information"     in ds
        assert "high_energy_theory"      in ds
        assert "quantum_mechanics"       in ds
        assert "bayesian_probability"    in ds
        assert "linear_algebra"          in ds
        assert "exoplanet_astrophysics"  in ds

    def test_every_domain_has_a_description(self):
        for name, d in load_domains().items():
            assert d.description, f"{name} missing description"

    def test_every_domain_pattern_compiles(self):
        """A typo in a regex would break the classifier at runtime; catch
        it at test time instead."""
        import re
        for name, d in load_domains().items():
            for p in d.latex_patterns:
                re.compile(p)
            for p in d.keyword_patterns:
                re.compile(p, re.IGNORECASE)

    def test_no_pattern_matches_empty_or_benign_text(self):
        """Regression: an earlier draft had r'\\|[A-Za-z]+\\|' as a 'norm'
        regex, which the regex engine read as 'empty OR letters\\|' and
        produced 45 matches in plain English. Catch this class of bug
        instead of letting it silently classify everything as 'norm'."""
        import re
        benign = "The quick brown fox jumps over the lazy dog. " * 3
        for name, d in load_domains().items():
            for p in d.latex_patterns:
                hits = re.compile(p).findall(benign)
                assert all(h for h in hits), (
                    f"{name}: pattern {p!r} matched empty strings -- likely "
                    "a regex bug (e.g. unescaped pipe meaning alternation)"
                )


# --- tier-1 classifier ------------------------------------------------------

@pytest.fixture
def domains():
    return load_domains()


class TestClassifierTier1:
    def test_qi_document_classifies_as_qi(self, domains):
        text = (
            r"Consider a density matrix $\rho$. The von Neumann entropy "
            r"$S(\rho) = -\text{Tr}(\rho \log \rho)$ is zero for pure states. "
            r"A POVM $\{E_i\}$ measures the system; for a Bell state "
            r"$\ket{\psi^+} = (\ket{00} + \ket{11})/\sqrt 2$ the reduced density "
            r"matrix is maximally mixed."
        )
        assert classify_tier1(text, domains) == "quantum_information"

    def test_hep_document_classifies_as_hep(self, domains):
        text = (
            r"The QED Lagrangian is $\mathcal{L} = -\frac{1}{4} F_{\mu\nu} "
            r"F^{\mu\nu} + \bar\psi(i\gamma^\mu \partial_\mu - m)\psi - e \bar\psi "
            r"\gamma^\mu A_\mu \psi$. The path integral $\int \mathcal{D}\phi$ "
            r"defines the partition function in this gauge theory."
        )
        assert classify_tier1(text, domains) == "high_energy_theory"

    def test_bayes_document_classifies_as_bayes(self, domains):
        text = (
            r"By Bayes' rule, the posterior $p(\theta | D) \propto p(D | \theta) "
            r"p(\theta)$. We use a conjugate Beta prior. NUTS in MCMC samples "
            r"the posterior efficiently."
        )
        assert classify_tier1(text, domains) == "bayesian_probability"

    def test_exoplanet_document_classifies_as_exo(self, domains):
        text = (
            r"The transit depth $\delta = (R_p/R_*)^2$ for an Earth-radius "
            r"planet around a star with $R_\odot$ and $T_{eff} = 5800$. "
            r"We detected the signal via radial velocity using HARPS-N."
        )
        assert classify_tier1(text, domains) == "exoplanet_astrophysics"

    def test_random_prose_returns_none(self, domains):
        assert classify_tier1(
            "The quick brown fox jumps over the lazy dog. " * 5,
            domains,
        ) is None

    def test_tied_top_two_returns_none(self):
        """When two domains have the same top hit count, the classifier
        bails so the (optional) LLM tie-breaker can handle it."""
        a = Domain("a", "A", latex_patterns=(r"\\foo",))
        b = Domain("b", "B", latex_patterns=(r"\\bar",))
        assert classify_tier1(r"\foo \bar", {"a": a, "b": b}) is None


# --- AcronymDetector --------------------------------------------------------

class TestAcronymDetector:
    def test_substitutes_a_known_acronym(self):
        d = AcronymDetector(acronyms={"POVM": "positive operator valued measure"})
        out = normalize_text(Pipeline([d]), "Measure with a POVM and record.")
        assert "positive operator valued measure" in out
        assert "POVM" not in out

    def test_pluralization(self):
        """Regression: 'POVMs' (plural) should expand to '...measures' so the
        spoken text still parses, not stay as 'POVMs' (which the TTS would
        spell P O V M S)."""
        d = AcronymDetector(acronyms={"POVM": "positive operator valued measure"})
        out = normalize_text(Pipeline([d]), "We compared two POVMs.")
        assert "positive operator valued measures" in out

    def test_case_sensitive_word_bounded(self):
        """A lowercase or sub-word match must NOT trigger. Otherwise common
        English words that happen to spell-match an acronym would be
        clobbered (e.g. 'gut' the word vs 'GUT' the grand-unified-theory)."""
        d = AcronymDetector(acronyms={"GUT": "grand unified theory"})
        out = normalize_text(Pipeline([d]), "I felt it in my gut, not a GUT.")
        # The word 'gut' is preserved; only 'GUT' is expanded.
        assert "gut" in out  # original word
        assert "grand unified theory" in out

    def test_longest_first(self):
        """'2-SRE' should expand to the longer entry, not be split into
        '2' + 'SRE'."""
        d = AcronymDetector(acronyms={
            "SRE":   "stabilizer Renyi entropy",
            "2-SRE": "two stabilizer Renyi entropy",
        })
        out = normalize_text(Pipeline([d]), "The 2-SRE is sub-additive.")
        assert "two stabilizer Renyi entropy" in out
        # And the bare 'SRE' rule did not also fire on the embedded substring.
        assert out.count("stabilizer Renyi entropy") == 1

    def test_pronunciations_are_case_insensitive(self):
        d = AcronymDetector(pronunciations={"qubit": "kew bit"})
        out = normalize_text(Pipeline([d]), "Each Qubit and qubit are kew bit.")
        # All three case variants get respelled.
        assert out.count("kew bit") == 3

    def test_empty_tables_leaves_text_alone(self):
        d = AcronymDetector()
        out = normalize_text(Pipeline([d]), "Nothing to see here.")
        assert out == "Nothing to see here."


# --- DocumentReader (offline path) -----------------------------------------

class TestDocumentReaderOffline:
    """With llm_factory=None, DocumentReader must work entirely offline:
    classify via tier-1 regex, read equations via the rule-based reader."""

    def test_classifies_then_reads_equation_offline(self, domains):
        text = (
            r"The QED Lagrangian $\mathcal{L} = -\frac{1}{4} F_{\mu\nu} "
            r"F^{\mu\nu} + \bar\psi(i\gamma^\mu \partial_\mu)\psi$ is the "
            r"starting point. With path integrals $\int \mathcal{D}\phi$..."
        )
        dr = DocumentReader(
            document_text=text,
            domains=domains,
            llm_factory=None,
            cache_dir=None,
        )
        assert dr.classify() == "high_energy_theory"
        # Equation reading falls back to the rule-based reader when there's
        # no LLM; output should be a non-empty string and not raise.
        out = dr.read_equation(r"E = mc^2")
        assert out and isinstance(out, str)

    def test_explicit_domain_skips_classification(self, domains):
        dr = DocumentReader(
            document_text="anything",
            domains=domains,
            llm_factory=None,
            explicit_domain="quantum_information",
        )
        assert dr.classify() == "quantum_information"

    def test_unclassifiable_doc_returns_none(self, domains):
        dr = DocumentReader(
            document_text="The quick brown fox jumps over the lazy dog. " * 3,
            domains=domains,
            llm_factory=None,
            cache_dir=None,
        )
        assert dr.classify() is None


# --- Reader integration -----------------------------------------------------

class TestReaderDomainIntegration:
    """End-to-end: Reader with domain= active produces utterances whose text
    has been through the AcronymDetector. We don't synthesize audio here --
    we just inspect the utterance stream."""

    @pytest.fixture
    def fake_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    def test_explicit_domain_expands_acronyms_in_utterances(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai", domain="quantum_information")
        out = list(r.utterances("A POVM measures the qubit."))
        full = " ".join(u.text for u in out)
        assert "positive operator valued measure" in full
        assert "kew bit" in full  # pronunciation override

    def test_auto_domain_picks_qi_from_content(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai", domain="auto")
        md = (
            r"# Notes" "\n\n"
            r"Density matrix $\rho$, POVM $\{E_i\}$, and a Bell state $\ket{\psi}$."
        )
        out = list(r.utterances(md))
        full = " ".join(u.text for u in out)
        assert "positive operator valued measure" in full

    def test_no_domain_passes_through_unchanged(self, fake_openai_key):
        """Backward-compat: Reader() without domain= behaves exactly as
        before -- no acronym substitution from any domain."""
        from mellifluous import Reader
        r = Reader(engine="openai")
        out = list(r.utterances("A POVM measures the qubit."))
        full = " ".join(u.text for u in out)
        assert "POVM" in full  # not expanded
        assert "positive operator valued measure" not in full

    def test_unknown_domain_name_raises(self, fake_openai_key):
        from mellifluous import Reader
        r = Reader(engine="openai", domain="not_a_real_domain")
        with pytest.raises(ValueError, match="unknown domain"):
            list(r.utterances("hello"))

    def test_auto_with_no_match_does_not_raise(self, fake_openai_key):
        """An unclassifiable document under domain='auto' should produce
        utterances normally (no acronym expansion, generic equation reader)
        rather than crash."""
        from mellifluous import Reader
        r = Reader(engine="openai", domain="auto")
        out = list(r.utterances("Just prose, nothing technical here."))
        assert any(u.text for u in out)
