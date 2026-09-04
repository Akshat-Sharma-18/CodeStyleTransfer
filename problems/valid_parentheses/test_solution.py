from solution import solve


def test_balanced():
    assert solve("([]{})") is True


def test_unbalanced_order():
    assert solve("([)]") is False


def test_unmatched_open():
    assert solve("(((") is False


def test_empty():
    assert solve("") is True


def test_ignores_other_chars():
    assert solve("a(b)c[d]e") is True
