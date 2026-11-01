from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Union

from .core import Encoding

_ENCODINGS_DIR = Path(__file__).parent / "encodings"
_lock = threading.Lock()
_cache: Dict[str, Encoding] = {}


def list_encoding_names() -> List[str]:
    """Names of the encodings bundled with the package."""
    if not _ENCODINGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in _ENCODINGS_DIR.glob("*.json"))


def get_encoding(name_or_path: Union[str, os.PathLike]) -> Encoding:
    """Loads a bundled encoding by name (e.g. "<variant>_32k") or any saved encoding file."""
    key = os.fspath(name_or_path)
    with _lock:
        if key in _cache:
            return _cache[key]
        path = _ENCODINGS_DIR / f"{key}.json"
        if not path.is_file():
            path = Path(key)
        if not path.is_file():
            raise ValueError(
                f"Unknown encoding {key!r}. Bundled encodings: {list_encoding_names()}; "
                "or pass the path of a file written by Encoding.save()."
            )
        enc = Encoding.load(path)
        _cache[key] = enc
        return enc
