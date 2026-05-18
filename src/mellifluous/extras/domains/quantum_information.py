"""Quantum information theory.

I am drafting these tables from general knowledge of the field. Audit
against your own papers and PR corrections; the entries most likely to be
wrong or incomplete are the acronyms (field jargon moves fast) and the
equation reader prompt's specific phrasing conventions.
"""
from ._base import Domain


DOMAIN = Domain(
    name="quantum_information",
    description=(
        "Quantum information theory: qubits, density matrices, entanglement, "
        "quantum channels, error correction, entropy measures, tomography."
    ),

    latex_patterns=(
        r"\\rho",
        r"\\sigma_[xyz]",
        r"\\text\{Tr\}_?",
        r"\\operatorname\{Tr\}",
        r"S\(\s*\\rho",                # von Neumann entropy
        r"\\otimes",
        r"\\langle\s*\\psi",
        r"\\ket\{",
        r"\\bra\{",
        r"\\braket\{",
        r"\\dagger",
        r"\|\s*\\psi\s*\\rangle",
        r"\|\s*\\phi\s*\\rangle",
        r"\\mathcal\{[EH]\}",           # channels, Hamiltonians
    ),
    keyword_patterns=(
        r"\bdensity matrix\b",
        r"\bvon Neumann entropy\b",
        r"\bquantum channel\b",
        r"\bentanglement entropy\b",
        r"\bquantum error correction\b",
        r"\bstabilizer\b",
        r"\bRenyi entropy\b",
        r"\bPOVM\b",
        r"\bBell state\b",
        r"\btomography\b",
    ),

    acronyms={
        "POVM":   "positive operator valued measure",
        "PVM":    "projective valued measure",
        "QEC":    "quantum error correction",
        "QECC":   "quantum error correcting code",
        "CPTP":   "completely positive trace preserving",
        "LOCC":   "local operations and classical communication",
        "GHZ":    "Greenberger Horne Zeilinger",
        "EPR":    "Einstein Podolsky Rosen",
        "SDP":    "semidefinite program",
        "MERA":   "multiscale entanglement renormalization ansatz",
        "PEPS":   "projected entangled pair states",
        "MPS":    "matrix product state",
        "MPO":    "matrix product operator",
        "VQE":    "variational quantum eigensolver",
        "QAOA":   "quantum approximate optimization algorithm",
        "QFT":    "quantum Fourier transform",
        "2-SRE":  "two stabilizer Renyi entropy",
        "SRE":    "stabilizer Renyi entropy",
        "CSS":    "Calderbank Shor Steane",
        "QKD":    "quantum key distribution",
        "BB84":   "B B eighty four",
    },

    pronunciations={
        "qubit":   "kew bit",
        "qubits":  "kew bits",
        "qudit":   "kew dit",
        "qudits":  "kew dits",
        "ansatz":  "ahn zats",
        "ansaetze": "ahn zet seh",
        "ket":     "ket",
        "bra":     "brah",
        "braket":  "brah ket",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from quantum information theory "
        "into a concise, natural English reading suitable for being spoken "
        "aloud by a text-to-speech system. You are reading the way a "
        "quantum-information researcher would explain the equation to a "
        "colleague at a whiteboard.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- Read bra-ket notation idiomatically: \\ket{\\psi} = 'ket psi', "
        "\\bra{\\phi} = 'bra phi', \\braket{\\phi|\\psi} = 'the inner "
        "product of phi and psi', \\ket{0}\\bra{1} = 'the outer product "
        "of zero and one'.\n"
        "- For tensor products, say 'tensor' (not 'otimes' or 'cross').\n"
        "- For density matrices: \\rho is 'rho', and operators on density "
        "matrices read in the style of 'trace of rho squared'.\n"
        "- For dagger, say 'dagger': U^\\dagger = 'U dagger'.\n"
        "- For partial traces, say 'partial trace over B' for \\text{Tr}_B.\n"
        "- Greek letters: spell them ('alpha', 'psi', 'rho', ...).\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
