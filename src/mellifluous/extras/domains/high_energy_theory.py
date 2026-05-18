"""High-energy physics / quantum field theory.

Drafted from general knowledge of QFT/HEP conventions. Audit; the
acronym list is small and conservative on purpose because HEP shorthand
varies a lot by subfield. The equation reader prompt instructs the LLM to
summarize Lagrangians by their parts (kinetic + interaction + mass) rather
than reading term-by-term, which matches how the user (an HEP-th friend)
asked to hear them.
"""
from ._base import Domain


DOMAIN = Domain(
    name="high_energy_theory",
    description=(
        "High-energy physics and quantum field theory: Lagrangians, gauge "
        "fields, Feynman diagrams, path integrals, scattering amplitudes."
    ),

    latex_patterns=(
        r"\\partial_\\mu",
        r"\\partial_\{",
        r"\\bar\{?\\psi",
        r"\\gamma\^\\mu",
        r"\\gamma_\\mu",
        r"g_\{\\mu\\?\\?nu\}",
        r"g\^\{\\mu\\?\\?nu\}",
        r"F_\{\\mu\\?\\?nu\}",
        r"F\^\{\\mu\\?\\?nu\}",
        r"A_\\mu",
        r"A\^\\mu",
        r"\\mathcal\{L\}",
        r"\\bar\\psi",
        r"D_\\mu",
        r"S\s*=\s*\\int d\^4",
        r"e\^\{iS",
        r"\\mathcal\{D\}\\phi",
        r"\\hbar",
    ),
    keyword_patterns=(
        r"\bLagrangian\b",
        r"\bgauge (?:field|invariance|theory)\b",
        r"\bFeynman (?:diagram|rule|propagator)\b",
        r"\bpath integral\b",
        r"\bscattering amplitude\b",
        r"\bWilson loop\b",
        r"\bChern.?Simons\b",
        r"\bspinor\b",
        r"\brenormalization\b",
        r"\beffective field theory\b",
        r"\bQED\b",
        r"\bQCD\b",
    ),

    acronyms={
        "QED":   "quantum electrodynamics",
        "QCD":   "quantum chromodynamics",
        "EFT":   "effective field theory",
        "SM":    "Standard Model",
        "BSM":   "beyond the Standard Model",
        "MSSM":  "minimal supersymmetric Standard Model",
        "SUSY":  "soosy",  # commonly pronounced; phonetic so the TTS gets it right
        "GUT":   "grand unified theory",
        "EWSB":  "electroweak symmetry breaking",
        "CFT":   "conformal field theory",
        "AdS":   "anti de Sitter",
        "CKM":   "Cabibbo Kobayashi Maskawa",
        "PMNS":  "Pontecorvo Maki Nakagawa Sakata",
        "RGE":   "renormalization group equation",
        "LO":    "leading order",
        "NLO":   "next to leading order",
        "NNLO":  "next to next to leading order",
        "LHC":   "Large Hadron Collider",
        "ATLAS": "ATLAS",
        "CMS":   "CMS",
    },

    pronunciations={
        "Lagrangian":   "luh grahn jee an",
        "Hamiltonian":  "hamil tonian",
        "ansatz":       "ahn zats",
        "eigenvalue":   "eye gen value",
        "Yang-Mills":   "yang mills",
        "Higgs":        "higs",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from high-energy physics / "
        "quantum field theory into a concise, natural English reading for "
        "a text-to-speech system. You are reading the way a working HEP "
        "theorist would explain the equation to a colleague.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- For Lagrangian densities, summarize by structural piece rather "
        "than reading term-by-term. Example: instead of reading every "
        "index on a QED Lagrangian, say 'the kinetic term for the Dirac "
        "field, minus the mass term, plus the gauge field kinetic term, "
        "minus the interaction vertex coupling psi-bar gamma-mu psi to "
        "A-mu.' Identify each piece by its physical role.\n"
        "- Treat repeated Lorentz indices as implicit Einstein summation; "
        "you do not need to spell out 'sum over mu'.\n"
        "- partial_mu reads as 'partial mu', \\bar\\psi as 'psi bar', "
        "F^{\\mu\\nu} as 'F mu nu', \\gamma^5 as 'gamma five'.\n"
        "- For path integrals, say 'the path integral over phi of e to "
        "the i S of phi over h-bar'.\n"
        "- For Feynman diagrams in formula form, name the topology: "
        "'a one-loop self-energy correction', 'a tree-level vertex'.\n"
        "- Greek letters: spell them. h-bar is 'h-bar'.\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
