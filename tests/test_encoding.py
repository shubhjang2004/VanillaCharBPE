import pickle
import random

import pytest

# Variant-independent tests: this file is identical in AksharaG2P, VanillaBPE-Char
# and VanillaBPE-Byte. The package under test is chosen in conftest.py.
from conftest import tok

TEXTS = [
    "",
    " ",
    "\n\n\t  \r\n",
    "भारत सरकार  नई\tदिल्ली\n\nमें है।",
    "हैदराबाद్‌లో",  # mixed scripts inside one word
    "తెలంగాణ రాష్ట్రం హైదరాబాద్‌లో ఉంది",  # ZWNJ: kept raw, dropped by normalization
    "ഞാൻ കേരളത്തിൽ",  # chillu letters
    # unseen languages
    "ગુજરાત સરકાર ગાંધીનગરમાં છે",
    "ಕರ್ನಾಟಕ ಸರ್ಕಾರ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಇದೆ",
    "ଓଡ଼ିଶା ସରକାର ଭୁବନେଶ୍ୱରରେ ଅଛି",
    "ਪੰਜਾਬ ਸਰਕਾਰ ਚੰਡੀਗੜ੍ਹ ਵਿੱਚ ਹੈ",
    "অসম চৰকাৰ",
    "ा starts with a matra; ््् viramas",
    "Hindi ०१२३ Bengali ০১২৩ Tamil ௧௨௩ ascii 0123 😀 emoji, 中文",
    "<|endoftext|> as plain text",
]


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(enc, raw_enc, text):
    norm = tok.normalize(text)
    assert enc.decode(enc.encode_ordinary(text)) == norm
    assert enc.decode(enc.encode(text, disallowed_special=())) == norm
    assert raw_enc.decode(raw_enc.encode_ordinary(text)) == text
    assert raw_enc.decode(raw_enc.encode(text, disallowed_special=())) == text


@pytest.mark.parametrize(
    "a, b",
    [
        ("\u0958", "\u0915\u093c"),  # precomposed / decomposed nukta
        ("\u09dc", "\u09a1\u09bc"),
        ("\u0d7b", "\u0d28\u0d4d"),  # chillus
        ("\u0d7c", "\u0d30\u0d4d"),
        ("\u09ce", "\u09a4\u09cd"),  # khanda ta
        ("\u0a2a\u0a70\u0a1c", "\u0a2a\u0a02\u0a1c"),  # tippi -> bindi
        ("\u0a2a\u0a71\u0a15\u0a3e", "\u0a2a\u0a15\u0a4d\u0a15\u0a3e"),  # addak doubles the next consonant
        ("\u0a38\u0a71\u0a5b\u0a3e", "\u0a38\u0a1c\u0a3c\u0a4d\u0a1c\u0a3c\u0a3e"),  # ... with its nukta
        ("\u0d15\u0d57", "\u0d15\u0d4c"),  # AU length mark
        ("\u0c26\u0c4d\u200c\u0c32\u0c4b", "\u0c26\u0c4d\u0c32\u0c4b"),  # ZWNJ dropped
        ("a\u00a0b\u00ad", "a b"),  # NFKC + soft hyphen
    ],
)
def test_normalize(enc, a, b):
    assert tok.normalize(a) == tok.normalize(b) == b
    assert enc.encode_ordinary(a) == enc.encode_ordinary(b)


def test_normalize_keeps_whitespace():
    text = "\u092d\u093e\u0930\u0924  \u0938\u0930\u0915\u093e\u0930\n\n\t\u09a8\u09a4\u09c1\u09a8\r\n"
    assert tok.normalize(text) == text


def test_round_trip_random(enc):
    rng = random.Random(0)
    pool = [chr(c) for c in range(0x0900, 0x0D80)] + list(" \n\tab1.,‌‍😀")
    for _ in range(500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        assert enc.decode(enc.encode_ordinary(text)) == tok.normalize(text)


def test_specials(enc):
    text = "भारत<|endoftext|>বাংলা"
    with pytest.raises(ValueError):
        enc.encode(text)
    ids = enc.encode(text, allowed_special="all")
    assert enc.eot_token in ids
    assert enc.decode(ids) == text
    assert enc.eot_token not in enc.encode_ordinary(text)


def test_batch(enc):
    assert enc.encode_ordinary_batch(TEXTS) == [enc.encode_ordinary(t) for t in TEXTS]
    assert enc.decode_batch(enc.encode_ordinary_batch(TEXTS)) == [tok.normalize(t) for t in TEXTS]
    assert enc.encode_batch(TEXTS, disallowed_special=()) == [enc.encode(t, disallowed_special=()) for t in TEXTS]


def test_decode_tokens_bytes(enc):
    text = "தமிழ் भारत"
    ids = enc.encode_ordinary(text)
    assert b"".join(enc.decode_tokens_bytes(ids)).decode() == text


def test_save_load_pickle(enc, tmp_path):
    path = tmp_path / "enc.json"
    enc.save(path)
    assert enc.normalization == tok.DEFAULT_NORMALIZATION
    for other in (tok.Encoding.load(path), tok.get_encoding(path), pickle.loads(pickle.dumps(enc))):
        assert other.normalization == enc.normalization
        assert other.n_vocab == enc.n_vocab
        for t in TEXTS:
            assert other.encode_ordinary(t) == enc.encode_ordinary(t)


def test_vocab_size(enc):
    assert enc.n_vocab <= 1200


def test_file_without_normalization_loads_raw(enc):
    d = enc.to_dict()
    del d["normalization"]  # files written by 0.1.x
    assert tok.Encoding.from_dict(d).normalization is None
