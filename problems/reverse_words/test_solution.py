from solution import solve


def test_basic():
    assert solve("the sky is blue") == "blue is sky the"


def test_extra_spaces():
    assert solve("  hello   world  ") == "world hello"


def test_single_word():
    assert solve("hello") == "hello"


def test_empty():
    assert solve("") == ""
