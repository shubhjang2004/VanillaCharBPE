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
    "తెలంగాణ రాష్ట్రం హైదరాబాద్‌లో ఉంది",  # ZWNJ must survive
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
def test_round_trip(enc, text):
    assert enc.decode(enc.encode_ordinary(text)) == text
    assert enc.decode(enc.encode(text, disallowed_special=())) == text


def test_round_trip_random(enc):
    rng = random.Random(0)
    pool = [chr(c) for c in range(0x0900, 0x0D80)] + list(" \n\tab1.,‌‍😀")
    for _ in range(500):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        assert enc.decode(enc.encode_ordinary(text)) == text


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
    assert enc.decode_batch(enc.encode_ordinary_batch(TEXTS)) == TEXTS


def test_decode_tokens_bytes(enc):
    text = "தமிழ் भारत"
    ids = enc.encode_ordinary(text)
    assert b"".join(enc.decode_tokens_bytes(ids)).decode() == text


def test_save_load_pickle(enc, tmp_path):
    path = tmp_path / "enc.json"
    enc.save(path)
    for other in (tok.Encoding.load(path), tok.get_encoding(path), pickle.loads(pickle.dumps(enc))):
        assert other.n_vocab == enc.n_vocab
        for t in TEXTS:
            assert other.encode_ordinary(t) == enc.encode_ordinary(t)


def test_vocab_size(enc):
    assert enc.n_vocab <= 1200
