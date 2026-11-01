from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import AbstractSet, Collection, Dict, List, Literal, Optional, Sequence, Tuple, Union

from . import _core
from .normalize import NORMALIZATIONS, normalize

SCRIPT_NAMES: Tuple[str, ...] = tuple(_core.SCRIPT_NAMES)
SHARED_BASE: int = _core.SHARED_BASE
SPECIFIC_BASE: int = _core.SPECIFIC_BASE
MARKER_BASE: int = _core.MARKER_BASE
N_FIXED_TOKENS: int = _core.N_FIXED_TOKENS
N_SHARED: int = _core.N_SHARED
N_MARKERS: int = _core.N_MARKERS
VARIANT: str = _core.VARIANT

FORMAT_NAME = VARIANT
FORMAT_VERSION = 1


def _unit_name(unit: int) -> str:
    """Readable name of a unit: shared phonemes by their Devanagari name."""
    if unit < SHARED_BASE:
        return f"<0x{unit:02X}>"
    if unit < SPECIFIC_BASE:
        ch = chr(0x0900 + unit - SHARED_BASE)
        name = unicodedata.name(ch, f"OFFSET 0x{unit - SHARED_BASE:02X}")
        return name.replace("DEVANAGARI ", "").replace("LETTER ", "").replace(" ", "_")
    ch = chr(0x0900 + unit - SPECIFIC_BASE)
    return f"{ch}({SCRIPT_NAMES[(unit - SPECIFIC_BASE) // 128]})"


class Encoding:
    """A BPE encoding over Indic units with a tiktoken-like API.

    Token ids, in order: 256 raw bytes; N_SHARED shared phonemes (aksharag2p
    only: one id for क ক ક க ಕ ...); N_MARKERS script markers (aksharag2p
    only: <|deva|>, <|beng|>, ...); script-specific characters; learned
    merges; special tokens.
    """

    def __init__(
        self,
        name: str,
        *,
        pat_str: str,
        specific_units: Sequence[int],
        merges: Sequence[Tuple[int, int]],
        special_tokens: Dict[str, int],
        metadata: Optional[dict] = None,
        normalization: Optional[str] = None,
    ):
        if normalization is not None and normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization {normalization!r}; known: {NORMALIZATIONS}")
        self.name = name
        self.normalization = normalization
        self._pat_str = pat_str
        self._specific_units = [int(u) for u in specific_units]
        self._merges = [(int(a), int(b)) for a, b in merges]
        self._special_tokens = dict(special_tokens)
        self.metadata = dict(metadata or {})
        self._core_bpe = _core.CoreBPE(
            pat_str, self._specific_units, self._merges, self._special_tokens
        )

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r} n_vocab={self.n_vocab}>"

    # ------------------------------------------------------------------ encode

    def normalize(self, text: str) -> str:
        """The text this encoding actually encodes (see `normalize.py`)."""
        return normalize(text, self.normalization)

    def encode_ordinary(self, text: str) -> List[int]:
        """Encodes text, ignoring special tokens."""
        return self._core_bpe.encode_ordinary(self.normalize(text))

    def encode(
        self,
        text: str,
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> List[int]:
        """Encodes text. Same special-token semantics as tiktoken: a special
        token string in the text raises unless it is allowed or explicitly not
        disallowed."""
        text = self.normalize(text)
        allowed = self._resolve_allowed(allowed_special)
        self._check_disallowed(text, allowed, disallowed_special)
        return self._core_bpe.encode(text, allowed)

    def encode_ordinary_batch(self, text: Sequence[str]) -> List[List[int]]:
        return self._core_bpe.encode_ordinary_batch([self.normalize(t) for t in text])

    def encode_batch(
        self,
        text: Sequence[str],
        *,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = "all",
    ) -> List[List[int]]:
        text = [self.normalize(t) for t in text]
        allowed = self._resolve_allowed(allowed_special)
        for t in text:
            self._check_disallowed(t, allowed, disallowed_special)
        return self._core_bpe.encode_batch(text, allowed)

    def _resolve_allowed(self, allowed_special) -> set:
        if allowed_special == "all":
            return set(self._special_tokens)
        return set(allowed_special)

    def _check_disallowed(self, text: str, allowed: set, disallowed_special) -> None:
        if disallowed_special == "all":
            disallowed = set(self._special_tokens) - allowed
        else:
            disallowed = set(disallowed_special)
        if not disallowed:
            return
        pat = "|".join(re.escape(t) for t in sorted(disallowed, key=len, reverse=True))
        match = re.search(pat, text)
        if match:
            raise ValueError(
                f"Encountered text corresponding to disallowed special token {match.group()!r}.\n"
                "If you want this text to be encoded as a special token, pass it to "
                "`allowed_special`. If you want it encoded as normal text, pass "
                "`disallowed_special=()`."
            )

    # ------------------------------------------------------------------ decode

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        return self._core_bpe.decode_bytes(list(tokens))

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes tokens to text. For any text, decode(encode(text)) == normalize(text)."""
        return self._core_bpe.decode_bytes(list(tokens)).decode("utf-8", errors=errors)

    def decode_batch(self, batch: Sequence[Sequence[int]], errors: str = "replace") -> List[str]:
        return [
            b.decode("utf-8", errors=errors)
            for b in self._core_bpe.decode_batch_bytes([list(t) for t in batch])
        ]

    def decode_single_token_bytes(self, token: int, script: Union[str, int] = "deva") -> bytes:
        """Bytes of one token out of context; shared phonemes are rendered in `script`."""
        if isinstance(script, str):
            script = SCRIPT_NAMES.index(script)
        return self._core_bpe.decode_single_token_bytes(token, script)

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> List[bytes]:
        """Per-token bytes, rendering each shared phoneme in its script context."""
        out, script = [], 0
        for t in tokens:
            if MARKER_BASE <= t < N_FIXED_TOKENS:
                script = t - MARKER_BASE
                out.append(b"")
            else:
                out.append(self._core_bpe.decode_single_token_bytes(t, script))
        return out

    # ----------------------------------------------------------- inspection

    def token_phonemes(self, token: int) -> List[str]:
        """The phoneme / byte units of a token, e.g. ['KA', 'VOWEL_SIGN_AA', 'MA']."""
        if MARKER_BASE <= token < N_FIXED_TOKENS:
            return [self.marker_name(token)]
        if token in self._special_tokens.values():
            return [next(k for k, v in self._special_tokens.items() if v == token)]
        units = self._core_bpe.token_units(token)
        if units is None:
            raise KeyError(token)
        return [_unit_name(u) for u in units]

    @staticmethod
    def marker_name(token: int) -> str:
        return f"<|{SCRIPT_NAMES[token - MARKER_BASE]}|>"

    @staticmethod
    def script_marker(script: str) -> int:
        """Token id of the script-switch marker for e.g. 'deva', 'taml'."""
        if not N_MARKERS:
            raise ValueError(f"{VARIANT} has no script markers")
        return MARKER_BASE + SCRIPT_NAMES.index(script)

    @property
    def n_vocab(self) -> int:
        return self._core_bpe.n_vocab()

    @property
    def special_tokens_set(self) -> set:
        return set(self._special_tokens)

    @property
    def eot_token(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    def is_special_token(self, token: int) -> bool:
        return token in self._special_tokens.values()

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "name": self.name,
            "pat_str": self._pat_str,
            "specific_units": self._specific_units,
            "merges": self._merges,
            "special_tokens": self._special_tokens,
            "metadata": self.metadata,
            "normalization": self.normalization,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Encoding":
        if d.get("format") != FORMAT_NAME:
            raise ValueError(f"not a {FORMAT_NAME} encoding file (format={d.get('format')!r})")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported {FORMAT_NAME} encoding version {d.get('version')}")
        return cls(
            d["name"],
            pat_str=d["pat_str"],
            specific_units=d["specific_units"],
            merges=[tuple(m) for m in d["merges"]],
            special_tokens=d["special_tokens"],
            metadata=d.get("metadata"),
            normalization=d.get("normalization"),
        )

    def save(self, path: Union[str, os.PathLike]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Encoding":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def __getstate__(self) -> dict:
        return self.to_dict()

    def __setstate__(self, state: dict) -> None:
        self.__init__(
            state["name"],
            pat_str=state["pat_str"],
            specific_units=state["specific_units"],
            merges=[tuple(m) for m in state["merges"]],
            special_tokens=state["special_tokens"],
            metadata=state.get("metadata"),
            normalization=state.get("normalization"),
        )
