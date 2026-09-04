from solution import solve


def test_mixed_case():
    assert solve("Hello World") == 3


def test_no_vowels():
    assert solve("xyz") == 0


def test_empty():
    assert solve("") == 0


def test_all_vowels():
    assert solve("aeiouAEIOU") == 10
