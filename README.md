# VanillaBPE-Char

GitHub: [shubhjang2004/VanillaCharBPE](https://github.com/shubhjang2004/VanillaCharBPE) · pip/import name: `vanillabpe_char` · sibling repos: [AksharaG2P](https://github.com/shubhjang2004/AksharaG2P), [VanillaBPE](https://github.com/shubhjang2004/VanillaBPE)

Character-level vanilla BPE baseline for Indic scripts (AksharaG2P without cross-script phoneme sharing).

Part of a three-way controlled comparison for the AksharaG2P paper:

| repo | base units | cross-script sharing |
|---|---|---|
| **AksharaG2P** | one unit per Indic character, shared across scripts by Unicode block offset (क = ক = ક = க = ಕ) + script markers | yes |
| **VanillaBPE-Char** | one unit per Indic character, separate per script | no |
| **VanillaBPE-Byte** | UTF-8 bytes (GPT/tiktoken style) | no |

**Base units:** every character in the nine Brahmic blocks (U+0900–U+0D7F) is
its own unit, so क (Devanagari) and ক (Bengali) are *different* units. Characters
seen fewer than `min_unit_frequency` times during training fall back to UTF-8
bytes. There is no cross-script sharing and there are no script markers.

This isolates the effect of AksharaG2P's cross-script phoneme mapping: the two
tokenizers use the same base granularity (one unit per Indic character), and
differ only in whether that unit is shared across scripts.

## Same pipeline as AksharaG2P

All three repos share the same Rust BPE core and trainer, the same
pre-tokenizer regex (GPT-4 style, keeping matras/virama/ZWNJ inside words),
the same script-boundary splitting, special tokens, training procedure and
Python API. **The only source difference is the `VARIANT` section of
`src/script.rs`** (plus variant-specific tests). Verify with:

```bash
diff -r AksharaG2P/src VanillaBPE-Char/src
diff -r AksharaG2P/python/aksharag2p VanillaBPE-Char/python/vanillabpe_char
diff AksharaG2P/tests/test_encoding.py VanillaBPE-Char/tests/test_encoding.py
```

## Usage

```python
import vanillabpe_char as tok

enc = tok.train({"hi": "hindi.txt", "bn": "bengali.txt"}, vocab_size=32_000, lines_per_file=500_000)
enc.save("vanillabpe_char_32k.json")

enc = tok.get_encoding("vanillabpe_char_32k.json")
ids = enc.encode("भारत सरकार नई दिल्ली में है।")
assert enc.decode(ids) == "भारत सरकार नई दिल्ली में है।"
```

Training script (same lines as the other two tokenizers):
`python scripts/train_corpus.py --base /content/drive/MyDrive`.

Decoding is lossless: `decode(encode(text)) == text` for any string.

## Install

```bash
pip install git+https://github.com/shubhjang2004/VanillaCharBPE   # builds from source; needs Rust (https://rustup.rs)
```

## Development

```bash
pip install maturin pytest
maturin develop --release
cargo test && pytest tests
```
