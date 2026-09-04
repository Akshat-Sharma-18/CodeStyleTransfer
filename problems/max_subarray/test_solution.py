from solution import solve


def test_mixed():
    assert solve([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6


def test_all_negative():
    assert solve([-3, -1, -2]) == -1


def test_single_element():
    assert solve([5]) == 5


def test_all_positive():
    assert solve([1, 2, 3, 4]) == 10
