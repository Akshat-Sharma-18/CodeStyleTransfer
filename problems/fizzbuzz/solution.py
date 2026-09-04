def solve(limit: int) -> list[str]:
    """Return FizzBuzz strings for 1..limit inclusive."""
    results = []
    for current_number in range(1, limit + 1):
        if current_number % 15 == 0:
            results.append("FizzBuzz")
        elif current_number % 3 == 0:
            results.append("Fizz")
        elif current_number % 5 == 0:
            results.append("Buzz")
        else:
            results.append(str(current_number))
    return results
