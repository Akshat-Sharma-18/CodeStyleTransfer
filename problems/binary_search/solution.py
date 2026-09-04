def solve(sorted_numbers: list[int], target: int) -> int:
    """Return the index of target in sorted_numbers, or -1 if absent."""
    low_index, high_index = 0, len(sorted_numbers) - 1
    while low_index <= high_index:
        middle_index = (low_index + high_index) // 2
        if sorted_numbers[middle_index] == target:
            return middle_index
        elif sorted_numbers[middle_index] < target:
            low_index = middle_index + 1
        else:
            high_index = middle_index - 1
    return -1
