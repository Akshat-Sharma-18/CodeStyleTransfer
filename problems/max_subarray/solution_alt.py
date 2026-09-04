"""Brute-force O(n^2) scan instead of Kadane's algorithm.

Same answers, different asymptotics and different structure.
"""


def solve(numbers):
    best = numbers[0]
    for start in range(len(numbers)):
        total = 0
        for end in range(start, len(numbers)):
            total = total + numbers[end]
            if total > best:
                best = total
    return best
