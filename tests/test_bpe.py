from src.bpe import BPETokenizer


def test_bpe_round_trip():
    text = "oscillator: omega=2.0 zeta=0.1 t=1.0 x=+0.5\n" * 50
    tok = BPETokenizer.train(text, vocab_size=128)
    enc = tok.encode(text)
    assert tok.decode(enc) == text


def test_bpe_compresses_repeats():
    text = "abab" * 100
    tok = BPETokenizer.train(text, vocab_size=260)  # 256 byte vocab + 4 merges
    enc = tok.encode(text)
    # Should compress repeated 'ab' pairs into one token.
    assert len(enc) < len(text)


def test_bpe_handles_unseen_chars():
    tok = BPETokenizer.train("aaa", vocab_size=258)
    # 'b' wasn't in training but we still know byte 'b'.
    enc = tok.encode("b")
    assert tok.decode(enc) == "b"
