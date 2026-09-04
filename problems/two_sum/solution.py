def solve(numbers: list[int], target: int) -> list[int]:
    """Return indices of the two numbers that add up to target."""
    seen_index_by_value = {}
    for current_index, current_value in enumerate(numbers):
        complement = target - current_value
        if complement in seen_index_by_value:
            return [seen_index_by_value[complement], current_index]
        seen_index_by_value[current_value] = current_index
    return []
