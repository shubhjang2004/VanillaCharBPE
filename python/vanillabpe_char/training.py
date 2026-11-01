from __future__ import annotations

import itertools
import os
import sys
import time
from typing import Dict, Iterable, Iterator, Optional, Sequence, Union

from . import _core
from .core import N_FIXED_TOKENS, VARIANT, Encoding
from .normalize import DEFAULT_NORMALIZATION, normalize

DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>", "<|pad|>")

PathLike = Union[str, os.PathLike]


def _read_lines(path: PathLike, max_lines: Optional[int]) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        yield from itertools.islice(f, max_lines)


def _batched(it: Iterable[str], n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _count(files, texts, lines_per_file, batch_size, pat_str, normalization, log):
    trainer = _core.Trainer(pat_str)
    sources = []
    if isinstance(files, dict):
        sources += [(str(k), _read_lines(p, lines_per_file)) for k, p in files.items()]
    elif files is not None:
        sources += [(os.fspath(p), _read_lines(p, lines_per_file)) for p in files]
    if texts is not None:
        sources.append(("<texts>", iter(texts)))
    if not sources:
        raise ValueError("pass `files` and/or `texts`")

    t0 = time.time()
    for label, lines in sources:
        n = 0
        for batch in _batched(lines, batch_size):
            trainer.feed([normalize(t, normalization) for t in batch])
            n += len(batch)
        log(f"[{VARIANT}] {label}: {n:,} lines | unique pieces so far {trainer.n_unique_pieces():,}")
    log(f"[{VARIANT}] counting done in {time.time() - t0:.1f}s")
    return trainer, [label for label, _ in sources]


def _finish(trainer, sources, vocab_size, special_tokens, min_frequency, min_unit_frequency,
            lines_per_file, pat_str, normalization, name, log) -> Encoding:
    n_ordinary = vocab_size - len(special_tokens)
    if n_ordinary <= N_FIXED_TOKENS:
        raise ValueError(f"vocab_size must exceed {N_FIXED_TOKENS + len(special_tokens)}")
    t0 = time.time()
    log(f"[{VARIANT}] {name}: learning merges for vocab {vocab_size:,} ...")

    def progress(done: int, total: int) -> None:
        rate = done / max(time.time() - t0, 1e-9)
        log(f"[{VARIANT}] {done:,}/{total:,} merges ({rate:,.0f}/s)")

    specific_units, merges = trainer.finish(
        n_ordinary, min_frequency, min_unit_frequency, progress if log is not _quiet else None
    )
    first_special = N_FIXED_TOKENS + len(specific_units) + len(merges)
    specials = {tok: first_special + i for i, tok in enumerate(special_tokens)}
    log(
        f"[{VARIANT}] {len(merges):,} merges, {len(specific_units)} script-specific units, "
        f"vocab {first_special + len(specials):,} in {time.time() - t0:.1f}s"
    )
    return Encoding(
        name,
        pat_str=pat_str,
        specific_units=specific_units,
        merges=merges,
        special_tokens=specials,
        metadata={
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "min_unit_frequency": min_unit_frequency,
            "lines_per_file": lines_per_file,
            "sources": sources,
        },
        normalization=normalization,
    )


def _quiet(*args) -> None:
    pass


def _logger(verbose: bool):
    return (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else _quiet


def train(
    files: Union[Sequence[PathLike], Dict[str, PathLike], None] = None,
    *,
    texts: Optional[Iterable[str]] = None,
    vocab_size: int = 32_000,
    lines_per_file: Optional[int] = None,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    min_frequency: int = 2,
    min_unit_frequency: int = 10,
    batch_size: int = 20_000,
    pat_str: str = _core.DEFAULT_PATTERN,
    normalization: Optional[str] = DEFAULT_NORMALIZATION,
    name: str = VARIANT,
    verbose: bool = True,
) -> Encoding:
    """Trains a corpus-level BPE encoding.

    Args:
        files: corpus files (a list, or a {lang: path} dict). Read line by
            line with newlines kept, so whitespace and line breaks are learned.
        texts: alternatively/additionally, an iterable of documents.
        vocab_size: total vocabulary size including special tokens.
        lines_per_file: read at most this many lines from each file.
        min_frequency: stop merging when the best pair is rarer than this.
        min_unit_frequency: script-specific signs (danda, avagraha, ...) rarer
            than this get no token of their own and fall back to bytes.
        normalization: applied to the corpus here and to every text the
            encoding encodes later (see `normalize.py`); None keeps text as is.
    """
    return train_many(
        files, texts=texts, vocab_sizes=[vocab_size], lines_per_file=lines_per_file,
        special_tokens=special_tokens, min_frequency=min_frequency,
        min_unit_frequency=min_unit_frequency, batch_size=batch_size, pat_str=pat_str,
        normalization=normalization, name_format=name, verbose=verbose,
    )[vocab_size]


def train_many(
    files: Union[Sequence[PathLike], Dict[str, PathLike], None] = None,
    *,
    texts: Optional[Iterable[str]] = None,
    vocab_sizes: Sequence[int] = (16_000, 32_000, 64_000, 128_000),
    lines_per_file: Optional[int] = None,
    special_tokens: Sequence[str] = DEFAULT_SPECIAL_TOKENS,
    min_frequency: int = 2,
    min_unit_frequency: int = 10,
    batch_size: int = 20_000,
    pat_str: str = _core.DEFAULT_PATTERN,
    normalization: Optional[str] = DEFAULT_NORMALIZATION,
    name_format: str = VARIANT + "_{k}k",
    verbose: bool = True,
) -> Dict[int, Encoding]:
    """Like `train`, but reads the corpus once and learns one encoding per
    vocab size. `name_format` may use {k} (size in thousands) and {size}."""
    log = _logger(verbose)
    trainer, sources = _count(files, texts, lines_per_file, batch_size, pat_str, normalization, log)
    return {
        size: _finish(
            trainer, sources, size, special_tokens, min_frequency, min_unit_frequency,
            lines_per_file, pat_str, normalization, name_format.format(k=size // 1000, size=size), log,
        )
        for size in vocab_sizes
    }
