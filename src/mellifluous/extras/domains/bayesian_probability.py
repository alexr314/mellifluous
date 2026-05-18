"""Bayesian probability and statistics.

Covers Bayes' rule, posteriors, likelihoods, conjugate priors, MCMC, etc.
The equation reader prompt steers toward 'the probability of X given Y'
phrasing rather than reading every bar and bracket.
"""
from ._base import Domain


DOMAIN = Domain(
    name="bayesian_probability",
    description=(
        "Bayesian probability and statistics: priors, posteriors, "
        "likelihoods, conditional probability, MCMC, conjugate analysis."
    ),

    latex_patterns=(
        r"P\([^)]*\|[^)]*\)",                    # P(A|B)
        r"p\([^)]*\|[^)]*\)",
        r"P\(\\theta",
        r"p\(\\theta",
        r"\\propto",
        r"\\sim\s+(?:N|\\mathcal\{N\}|Beta|Gamma|Bernoulli|Poisson|Dirichlet|Uniform)",
        r"\\mathcal\{N\}\(",
        r"\\text\{Beta\}",
        r"\\text\{Gamma\}",
        r"\\mathbb\{E\}",                        # expectation
        r"\\mathbb\{P\}",
        r"\\hat\{\\theta\}",                     # estimator
        r"\\arg\\?max",
        r"\\arg\\?min",
    ),
    keyword_patterns=(
        r"\bposterior\b",
        r"\bprior\b",
        r"\blikelihood\b",
        r"\bmarginal likelihood\b",
        r"\bBayes(?:'|ian)?\b",
        r"\bconjugate prior\b",
        r"\bMCMC\b",
        r"\bMetropolis(?:.Hastings)?\b",
        r"\bGibbs sampling\b",
        r"\bcredible interval\b",
        r"\bMAP estimate\b",
    ),

    acronyms={
        "MAP":   "maximum a posteriori",
        "MLE":   "maximum likelihood estimate",
        "MCMC":  "Markov chain Monte Carlo",
        "HMC":   "Hamiltonian Monte Carlo",
        "NUTS":  "no U turn sampler",
        "ELBO":  "evidence lower bound",
        "KL":    "K L divergence",
        "VI":    "variational inference",
        "ABC":   "approximate Bayesian computation",
        "BIC":   "Bayesian information criterion",
        "DIC":   "deviance information criterion",
        "WAIC":  "widely applicable information criterion",
        "BMA":   "Bayesian model averaging",
        "IID":   "independent and identically distributed",
    },

    pronunciations={
        "Bayes":       "bayz",
        "Bayesian":    "bayz ee an",
        "posterior":   "pos teer ee or",
        "prior":       "pry or",
        "Dirichlet":   "deer ik lay",
        "Gaussian":    "gow see an",
        "Bernoulli":   "ber noo lee",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from Bayesian probability and "
        "statistics into a concise, natural English reading suitable for "
        "being spoken aloud by a text-to-speech system.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- P(A|B) reads as 'the probability of A given B', NOT 'P open "
        "paren A bar B'.\n"
        "- p(theta|D) is 'the posterior of theta given the data D'.\n"
        "- \\propto reads as 'is proportional to'.\n"
        "- X \\sim N(mu, sigma^2) reads as 'X is distributed as a normal "
        "with mean mu and variance sigma squared'.\n"
        "- \\mathbb{E}[X|Y] is 'the expectation of X given Y'.\n"
        "- arg max / arg min: 'arg max over theta of' is fine.\n"
        "- For Bayes' rule, identify the parts by name when natural: "
        "'posterior proportional to likelihood times prior'.\n"
        "- Greek letters: spell them.\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
