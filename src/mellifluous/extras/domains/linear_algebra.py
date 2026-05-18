"""Linear algebra and matrix computation.

Pretty universal so the patterns are quite generic; the classifier will
rarely pick this if a more specific physics/QI domain is also a candidate
in the document. That's fine -- the more specific domain's reader prompt
will usually do a good job on matrix expressions in context anyway.
"""
from ._base import Domain


DOMAIN = Domain(
    name="linear_algebra",
    description=(
        "Linear algebra and matrix computation: matrices, vectors, "
        "eigenvalues, decompositions, tensor products."
    ),

    latex_patterns=(
        r"\\det\b",
        r"\\text\{tr\}",
        r"\\operatorname\{tr\}",
        r"\\text\{rank\}",
        r"A\^T\b",
        r"A\^\{-1\}",
        r"\\mathbf\{[A-Za-z]\}",
        r"\\boldsymbol\{[A-Za-z]\}",
        r"\\lambda_\{?i\}?",
        r"\\Sigma",
        r"\\otimes",
        r"\\oplus",
        # Norms: LaTeX writes them as \| ... \|. In a Python raw regex we
        # need r"\\\|" to match the literal backslash + pipe.
        r"\\\|[A-Za-z]+\\\|",
        r"\\langle\s*[A-Za-z],\s*[A-Za-z]\s*\\rangle",  # inner products
    ),
    keyword_patterns=(
        r"\beigenvalue\b",
        r"\beigenvector\b",
        r"\bsingular value\b",
        r"\bSVD\b",
        r"\bdiagonali[sz]ation\b",
        r"\borthogonal matrix\b",
        r"\bspectral (?:theorem|decomposition)\b",
        r"\b(?:positive (?:semi)?definite)\b",
        r"\bdeterminant\b",
        r"\btrace\b",
        r"\bbasis\b",
        r"\binner product space\b",
    ),

    acronyms={
        "SVD":  "singular value decomposition",
        "QR":   "Q R decomposition",
        "LU":   "L U decomposition",
        "PCA":  "principal component analysis",
        "PSD":  "positive semidefinite",
        "PD":   "positive definite",
        "SPD":  "symmetric positive definite",
        "ONB":  "orthonormal basis",
        "rref": "reduced row echelon form",
    },

    pronunciations={
        "eigenvalue":   "eye gen value",
        "eigenvalues":  "eye gen values",
        "eigenvector":  "eye gen vector",
        "eigenvectors": "eye gen vectors",
        "eigenbasis":   "eye gen basis",
        "Hermitian":    "her mish un",
        "unitary":      "you ni tare ee",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from linear algebra into a "
        "concise, natural English reading suitable for being spoken aloud "
        "by a text-to-speech system.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- A^T reads as 'A transpose'. A^{-1} is 'A inverse'. A^\\dagger "
        "is 'A dagger'.\n"
        "- \\det(A) is 'the determinant of A'.\n"
        "- \\text{tr}(A) or \\operatorname{tr}(A) is 'the trace of A'.\n"
        "- For decompositions, name them: A = U Sigma V^T is 'A equals U "
        "Sigma V transpose, the singular value decomposition'.\n"
        "- Matrix multiplication: read juxtaposition as 'times' or 'into', "
        "whichever sounds more natural.\n"
        "- For norms, \\|x\\|_2 is 'the two-norm of x', \\|x\\|_\\infty "
        "is 'the infinity norm of x'.\n"
        "- \\otimes is 'tensor', \\oplus is 'direct sum'.\n"
        "- Greek letters: spell them.\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
