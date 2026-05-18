"""Exoplanet astrophysics: transits, radial velocity, atmospheres.

The most specialized domain in this library. The acronym table covers
mission names (TESS, JWST) and standard astrophysical quantities. The
equation reader prompt knows about transit notation, stellar parameters,
and the M_oplus / R_sun unit conventions.

Audit: the unit-symbol substitutions (M_\\oplus, R_\\odot, M_\\jupiter)
benefit from being read as 'Earth masses' etc. rather than letter-by-letter,
but LaTeX doesn't standardize those; we lean on the equation reader prompt
to get this right.
"""
from ._base import Domain


DOMAIN = Domain(
    name="exoplanet_astrophysics",
    description=(
        "Exoplanet astrophysics: transit photometry, radial velocity, "
        "stellar characterization, planetary atmospheres, orbital dynamics."
    ),

    latex_patterns=(
        r"M_\\oplus",
        r"R_\\oplus",
        r"M_\\odot",
        r"R_\\odot",
        r"L_\\odot",
        r"M_\{?\\rm\s*[Jj]up",
        r"R_\{?\\rm\s*[Jj]up",
        r"T_\{?\\rm\s*eff",
        r"T_\\text\{eff\}",
        r"\\log\s*g",
        r"\\delta\s*=\s*\(R_p",                  # transit depth
        r"R_p/R_\\?[*\\star]",
        r"a/R_\\?[*\\star]",
        r"\\text\{\\AA\}",
        r"\\AA",                                 # Angstrom
    ),
    keyword_patterns=(
        r"\bexoplanet\b",
        r"\btransit (?:depth|duration|photometry|method)\b",
        r"\bradial velocity\b",
        r"\bRV (?:method|signal|measurement)\b",
        r"\blight curve\b",
        r"\bspectroscop(?:y|ic)\b",
        r"\b(?:hot|warm) Jupiter\b",
        r"\bsuper.?Earth\b",
        r"\bhabitable zone\b",
        r"\beffective temperature\b",
        r"\bmetallicity\b",
        r"\blimb darkening\b",
    ),

    acronyms={
        "TESS":   "T E S S",
        "JWST":   "James Webb Space Telescope",
        "HST":    "Hubble Space Telescope",
        "Kepler": "Kepler",
        "PLATO":  "PLATO",
        "ARIEL":  "ARIEL",
        "CHEOPS": "CHEOPS",
        "RV":     "radial velocity",
        "SNR":    "signal to noise ratio",
        "BLS":    "box-fitting least squares",
        "GLS":    "generalized Lomb-Scargle",
        "TTV":    "transit timing variation",
        "TDV":    "transit duration variation",
        "BJD":    "barycentric Julian date",
        "AU":     "astronomical unit",
        "ppm":    "parts per million",
        "FAP":    "false alarm probability",
        "SED":    "spectral energy distribution",
        "MCMC":   "Markov chain Monte Carlo",
    },

    pronunciations={
        "exoplanet":     "exo planet",
        "exoplanets":    "exo planets",
        "Kepler":        "kep ler",
        "barycentric":   "barry centric",
        "ephemeris":     "ef em ur is",
        "Doppler":       "dop ler",
        "metallicity":   "metal iss it ee",
    },

    equation_reader_prompt=(
        "You convert LaTeX math expressions from exoplanet astrophysics "
        "into a concise, natural English reading suitable for being spoken "
        "aloud by a text-to-speech system.\n\n"
        "Rules:\n"
        "- Output ONLY the spoken text. No quotes, no LaTeX, no commentary.\n"
        "- Unit subscripts read as units, not letters: M_\\oplus is 'Earth "
        "masses', R_\\oplus is 'Earth radii', M_\\odot is 'solar masses', "
        "R_\\odot is 'solar radii', L_\\odot is 'solar luminosities', "
        "M_{Jup} is 'Jupiter masses'.\n"
        "- T_{eff} is 'effective temperature', \\log g is 'log g' "
        "(astronomers say it that way).\n"
        "- Transit depth: \\delta = (R_p/R_*)^2 reads 'the transit depth "
        "delta equals the planet radius over star radius, squared'.\n"
        "- a/R_* is 'a over R-star' (the scaled semi-major axis).\n"
        "- For RV semi-amplitude K = ..., name what it is: 'the radial "
        "velocity semi-amplitude'.\n"
        "- Wavelengths in Angstroms: \\lambda = 5000 \\AA is '5000 "
        "Angstroms'.\n"
        "- Greek letters: spell them.\n"
        "- Do not announce 'the equation is...' or similar.\n"
    ),
)
