"""Corpus-level BPE tokenizer for Indic scripts with a tiktoken-style API.

`VARIANT` says which tokenizer this is (aksharag2p / vanillabpe_char / vanillabpe_byte).
"""

from .core import SCRIPT_NAMES, VARIANT, Encoding
from .registry import get_encoding, list_encoding_names
from .training import train, train_many

__all__ = ["Encoding", "SCRIPT_NAMES", "VARIANT", "get_encoding", "list_encoding_names", "train", "train_many"]
__version__ = "0.1.0"
