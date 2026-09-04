from solution import solve


def test_overlapping():
    assert solve([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]


def test_touching():
    assert solve([[1, 4], [4, 5]]) == [[1, 5]]


def test_no_overlap():
    assert solve([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]


def test_empty():
    assert solve([]) == []


def test_unsorted_input():
    assert solve([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
