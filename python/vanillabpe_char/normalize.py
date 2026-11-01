"""Text normalization applied before training and encoding.

"nfkc_indic_v1" (the default for newly trained encodings):

1. NFKC, so that precomposed and decomposed spellings (क़ vs क + ़) are one
   sequence;
2. drops invisible format characters (ZWNJ, ZWJ, ZWSP, LRM, RLM, BOM, soft
   hyphen), as the paper's tokenizer does;
3. rewrites script-only signs into their sequence of ordinary letters, so they
   can share units with the other scripts:
       Malayalam chillus ൺ ൻ ർ ൽ ൾ ൿ ൔ ൕ ൖ  ->  consonant + virama
       Bengali khanda ta ৎ                   ->  ত + virama
       Gurmukhi tippi ੰ                      ->  bindi ਂ (anusvara)
       Gurmukhi addak ੱ + consonant          ->  consonant + virama + consonant
       Malayalam AU length mark ൗ            ->  AU sign ൌ

The same function is applied by all three packages (aksharag2p,
vanillabpe_char, vanillabpe_byte), so the comparison stays fair. Decoding
returns the normalized text: decode(encode(t)) == normalize(t).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

DEFAULT_NORMALIZATION = "nfkc_indic_v1"
NORMALIZATIONS = (DEFAULT_NORMALIZATION,)

_STRIP = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad"))

_REWRITES = str.maketrans({
    # Malayalam chillus -> consonant + virama (U+0D4D)
    "\u0d7a": "\u0d23\u0d4d", "\u0d7b": "\u0d28\u0d4d", "\u0d7c": "\u0d30\u0d4d",
    "\u0d7d": "\u0d32\u0d4d", "\u0d7e": "\u0d33\u0d4d", "\u0d7f": "\u0d15\u0d4d",
    "\u0d54": "\u0d2e\u0d4d", "\u0d55": "\u0d2f\u0d4d", "\u0d56": "\u0d34\u0d4d",
    # Bengali khanda ta -> ta + virama
    "\u09ce": "\u09a4\u09cd",
    # Gurmukhi tippi -> bindi
    "\u0a70": "\u0a02",
    # Malayalam AU length mark -> AU vowel sign (after NFKC, any e-sign +
    # length mark is already composed, so a remaining mark stands for AU)
    "\u0d57": "\u0d4c",
})

# Gurmukhi addak doubles the next consonant (with its nukta, if any)
_ADDAK = re.compile("\u0a71([\u0a15-\u0a39\u0a5c][\u0a3c]?)")


def normalize(text: str, normalization: Optional[str] = DEFAULT_NORMALIZATION) -> str:
    """Normalizes `text`; `normalization=None` returns it unchanged."""
    if normalization is None:
        return text
    if normalization != DEFAULT_NORMALIZATION:
        raise ValueError(f"unknown normalization {normalization!r}; known: {NORMALIZATIONS}")
    text = unicodedata.normalize("NFKC", text).translate(_STRIP).translate(_REWRITES)
    if "\u0a71" in text:
        text = _ADDAK.sub("\\1\u0a4d\\1", text)
    return text
