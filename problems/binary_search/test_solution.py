from solution import solve


def test_found_middle():
    assert solve([1, 3, 5, 7, 9], 5) == 2


def test_found_edges():
    assert solve([1, 3, 5, 7, 9], 1) == 0
    assert solve([1, 3, 5, 7, 9], 9) == 4


def test_not_found():
    assert solve([1, 3, 5, 7, 9], 4) == -1


def test_empty():
    assert solve([], 1) == -1
