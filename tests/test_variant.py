"""Tests specific to VanillaBPE-Char: per-script characters, no sharing, no markers."""

import pytest

from conftest import tok


def test_variant():
    assert tok.VARIANT == "vanillabpe_char"


def test_no_cross_script_sharing(enc):
    hi = enc.encode_ordinary(" सरकार")
    bn = enc.encode_ordinary(" সরকার")
    assert len(hi) == len(bn) == 1  # each learned as one word token ...
    assert hi != bn  # ... but separately per script


def test_no_script_markers(enc):
    with pytest.raises(ValueError):
        enc.script_marker("deva")


def test_unseen_script_falls_back_to_bytes(enc):
    # Gujarati characters were never seen in training, so they have no unit token
    ids = enc.encode_ordinary("ક")
    assert len(ids) == 3 and all(t < 256 for t in ids)
