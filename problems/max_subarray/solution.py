def solve(numbers: list[int]) -> int:
    """Return the largest sum of any contiguous subarray (Kadane's algorithm)."""
    best_sum = numbers[0]
    running_sum = numbers[0]
    for value in numbers[1:]:
        running_sum = max(value, running_sum + value)
        best_sum = max(best_sum, running_sum)
    return best_sum
