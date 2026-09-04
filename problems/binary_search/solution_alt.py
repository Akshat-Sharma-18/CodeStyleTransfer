"""Linear scan instead of binary search -- same contract, different algorithm.

Passes every binary_search test while sharing almost no structure with the
real solution. Exactly the kind of output a test-gate-only eval would wave
through.
"""


def solve(sorted_numbers, target):
    for index, value in enumerate(sorted_numbers):
        if value == target:
            return index
    return -1
