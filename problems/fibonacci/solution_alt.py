"""A deliberately DIFFERENT algorithm with identical behavior.

Used to test content-preservation metrics: this passes fibonacci's tests just
as well as the iterative version, so any eval that scores only "did it pass"
cannot tell the two apart. A faithful style rewrite of the iterative solution
should look like the iterative solution -- not like this.
"""


def solve(n):
    memo = {0: 0, 1: 1}

    def helper(k):
        if k in memo:
            return memo[k]
        memo[k] = helper(k - 1) + helper(k - 2)
        return memo[k]

    return helper(n)
