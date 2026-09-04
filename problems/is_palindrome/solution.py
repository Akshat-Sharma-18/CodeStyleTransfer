def solve(text: str) -> bool:
    """Check whether text reads the same forwards and backwards, ignoring case and spaces."""
    normalized_characters = [character.lower() for character in text if character.isalnum()]
    return normalized_characters == normalized_characters[::-1]
