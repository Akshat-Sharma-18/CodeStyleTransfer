from solution import solve


def test_small():
    assert solve(5) == ["1", "2", "Fizz", "4", "Buzz"]


def test_fizzbuzz_at_15():
    assert solve(15)[-1] == "FizzBuzz"


def test_one():
    assert solve(1) == ["1"]


def test_length():
    assert len(solve(30)) == 30
