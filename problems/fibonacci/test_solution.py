from solution import solve


def test_base_cases():
    assert solve(0) == 0
    assert solve(1) == 1


def test_small():
    assert solve(5) == 5


def test_larger():
    assert solve(10) == 55
