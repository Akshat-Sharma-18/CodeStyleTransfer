def solve(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed: fib(0) = 0, fib(1) = 1)."""
    previous_value, current_value = 0, 1
    for _ in range(n):
        previous_value, current_value = current_value, previous_value + current_value
    return previous_value
