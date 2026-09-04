from solution import solve


def test_basic():
    assert solve([2, 7, 11, 15], 9) == [0, 1]


def test_no_match():
    assert solve([1, 2, 3], 100) == []


def test_later_pair():
    assert solve([3, 2, 4], 6) == [1, 2]


def test_duplicates():
    assert solve([3, 3], 6) == [0, 1]
