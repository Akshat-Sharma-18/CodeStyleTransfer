def solve(text: str) -> int:
    """Count vowel characters (a, e, i, o, u) in text, case-insensitive."""
    vowels = set("aeiouAEIOU")
    count = 0
    for character in text:
        if character in vowels:
            count += 1
    return count
