import pytest

import vanillabpe_char as tok  # the only line that differs between the three repos

CORPUS = [
    "भारत सरकार नई दिल्ली में है। मुंबई एक बड़ा शहर है।\n",
    "मुंबई महाराष्ट्राची राजधानी आहे.\n",
    "বাংলাদেশ সরকার ঢাকায় আছে। কাজ করো।\n",
    "தமிழ்நாடு அரசு சென்னையில் உள்ளது.\n",
    "తెలంగాణ రాష్ట్రం హైదరాబాద్‌లో ఉంది.\n",
    "കേരള സർക്കാർ തിരുവനന്തപുരത്ത് ആണ്. മലയാളം ഒരു സുന്ദര ഭാഷയാണ്.\n",
    "The government of India, est. 1947 — 140 crore people!\n",
]


@pytest.fixture(scope="session")
def enc() -> tok.Encoding:
    return tok.train(texts=CORPUS * 30, vocab_size=1200, min_unit_frequency=1, verbose=False)


@pytest.fixture(scope="session")
def raw_enc() -> tok.Encoding:
    """Same corpus, no normalization: encoding is exactly lossless."""
    return tok.train(texts=CORPUS * 30, vocab_size=1200, min_unit_frequency=1, normalization=None, verbose=False)
