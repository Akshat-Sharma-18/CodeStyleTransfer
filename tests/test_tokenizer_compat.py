"""The CodeT5 tokenizer workaround has to behave like the tokenizer it replaces.

These are network-dependent (they pull vocab files from the Hub) and skip when
it is unavailable, but they are worth having: the EOS bug they pin cost a full
fine-tune. Nothing crashed and no test failed -- the model simply never learned
where to stop, and it only showed up as a mediocre eval score.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODEL = "Salesforce/codet5p-220m"


@pytest.fixture(scope="module")
def tokenizer():
    from model.tokenizer_compat import load_tokenizer

    try:
        return load_tokenizer(MODEL)
    except Exception as exc:  # noqa: BLE001 - offline or Hub down
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_encoding_terminates_with_eos(tokenizer):
    """Without this, seq2seq labels carry no EOS and the model never stops."""
    ids = tokenizer("def f(x):\n    return x")["input_ids"]
    assert ids[-1] == tokenizer.eos_token_id


def test_special_tokens_resolve_to_vocab_ids(tokenizer):
    for token in ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]:
        assert tokenizer.convert_tokens_to_ids(token) is not None
        assert tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id or token == "<unk>"


def test_round_trips_python_source_exactly(tokenizer):
    source = "def solve(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total"
    decoded = tokenizer.decode(tokenizer(source)["input_ids"], skip_special_tokens=True)
    assert decoded == source
