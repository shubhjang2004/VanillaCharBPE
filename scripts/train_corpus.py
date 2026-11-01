"""Train corpus-level encodings for several vocab sizes (reads the corpus once).

    pip install maturin && maturin develop --release
    python scripts/train_corpus.py --base /content/drive/MyDrive --sizes 32000 64000

Identical in AksharaG2P, VanillaBPE-Char and VanillaBPE-Byte except for the
import line, so all three tokenizers are trained on exactly the same lines.
"""

import argparse
import os

import vanillabpe_char as tok  # the only line that differs between the three repos

LANGS = {
    "hi": "hindi", "mr": "marathi", "bn": "bengali",
    "ta": "tamil", "te": "telugu", "ml": "malayalam",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/content/drive/MyDrive")
    ap.add_argument("--sizes", type=int, nargs="+", default=[16_000, 32_000, 64_000, 128_000])
    ap.add_argument("--lines-per-lang", type=int, default=500_000)
    ap.add_argument("--min-unit-frequency", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = {lang: f"{args.base}/corpus/{name}_corpus.txt" for lang, name in LANGS.items()}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing corpus files: {missing}")
    out_dir = args.out or f"{args.base}/tokenizers/{tok.VARIANT}"
    os.makedirs(out_dir, exist_ok=True)

    encodings = tok.train_many(
        files,
        vocab_sizes=args.sizes,
        lines_per_file=args.lines_per_lang,
        min_unit_frequency=args.min_unit_frequency,
    )
    for enc in encodings.values():
        enc.save(f"{out_dir}/{enc.name}.json")
        print(f"saved {out_dir}/{enc.name}.json  ({enc.n_vocab:,} tokens)")
