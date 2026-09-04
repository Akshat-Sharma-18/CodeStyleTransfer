def solve(sentence: str) -> str:
    """Reverse the order of words in sentence, collapsing extra whitespace."""
    words = sentence.split()
    return " ".join(reversed(words))
