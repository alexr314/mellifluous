"""Quantum mechanics (non-relativistic, bra-ket).

A more general QM domain than quantum_information: covers Schrodinger,
operators, perturbation theory, scattering -- the textbook-quantum side.
Some overlap with quantum_information is intentional; the classifier
picks whichever scores higher.
"""
from ._base import Domain


DOMAIN = Domain(
    name="quantum_mechanics",
    description=(
        "Quantum mechanics: Schrodinger equation, operators, bra-ket "
        "notation, perturbation theory, angular momentum, scattering."
    ),

    latex_patterns=(
        r"\\hat\{H\}",
        r"\\hat\{[A-Za-z]\}",
        r"i\\hbar\s*\\frac\{\\partial",          # iħ ∂/∂t
        r"\\Psi\(",
        r"\\psi\(",
        r"H\s*\\ket",
        r"\\langle\s*[A-Za-z\\]+\s*\|",
        r"\\hbar",
        r"\\nabla\^2",
        r"[\\hat]?\s*p\s*=\s*-i\\hbar",
        r"\\hat\{p\}",
        r"\\hat\{x\}",
    ),
    keyword_patterns=(
        r"\bSchr(?:o|ö)dinger\b",
        r"\bwave ?function\b",
        r"\bperturbation theory\b",
        r"\bcommutator\b",
        r"\bangular momentum\b",
        r"\bscattering (?:amplitude|cross section)\b",
        r"\beigenstate\b",
        r"\bobservable\b",
        r"\bharmonic oscillator\b",
        r"\bhydrogen atom\b",
    ),

    acronyms={
        "WKB":  "W K B",
        "JWKB": "J W K B",
        "EPR":  "Einstein Podolsky Rosen",
    },

    pronunciations={
        "Schrodinger":   "shro din ger",
        "Schroedinger":  "shro din ger",
        "Schrödinger":   "shro din ger",
        "Heisenberg":    "high zen berg",
        "Hamiltonian":   "hamil tonian",
        "eigenstate":    "eye gen state",
        "eigenstates":   "eye gen states",
        "eigenvalue":    "eye gen value",
        "eigenvalues":   "eye gen values",
        "Hermitian":     "her mish un",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from quantum mechanics into a "
        "concise, natural English reading suitable for being spoken aloud "
        "by a text-to-speech system.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- Read bra-ket idiomatically: \\ket{\\psi} = 'ket psi', "
        "\\bra{\\phi} = 'bra phi', \\braket{\\phi|\\psi} = 'phi bra ket psi' "
        "or 'the inner product of phi and psi'. Use 'expectation value of' "
        "for \\langle \\hat{O} \\rangle.\n"
        "- Hatted operators are read with 'hat': \\hat{H} = 'H hat'.\n"
        "- For commutators [A, B], say 'the commutator of A and B' or "
        "'A B minus B A' depending on which is more natural in context.\n"
        "- iħ ∂/∂t |\\psi> = H |\\psi> reads as 'i h-bar partial of psi "
        "with respect to t equals H hat psi'.\n"
        "- Greek letters: spell them. h-bar is 'h-bar'.\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
