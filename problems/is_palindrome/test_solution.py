from solution import solve


def test_simple_palindrome():
    assert solve("racecar") is True


def test_with_spaces_and_case():
    assert solve("A man a plan a canal Panama") is True


def test_not_palindrome():
    assert solve("hello") is False


def test_empty_string():
    assert solve("") is True
