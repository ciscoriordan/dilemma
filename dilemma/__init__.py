"""Dilemma - diachronic Greek lemmatizer.

    from dilemma import Dilemma

    d = Dilemma()
    d.lemmatize("ἔφατ̓")  # -> "φημί"

See README.md at the repo root for full docs.
"""

from .core import (
    Dilemma,
    LemmaCandidate,
    LookupDB,
    to_monotonic,
    grave_to_acute,
    strip_accents,
)
from ._download import download as download_data

__all__ = [
    "Dilemma",
    "LemmaCandidate",
    "LookupDB",
    "to_monotonic",
    "grave_to_acute",
    "strip_accents",
    "download_data",
]
