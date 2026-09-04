def solve(text: str) -> bool:
    """Check whether brackets in text are balanced and correctly nested."""
    closing_for_opening = {"(": ")", "[": "]", "{": "}"}
    opening_brackets = set(closing_for_opening.keys())
    closing_brackets = set(closing_for_opening.values())

    stack = []
    for character in text:
        if character in opening_brackets:
            stack.append(character)
        elif character in closing_brackets:
            if not stack or closing_for_opening[stack.pop()] != character:
                return False
    return not stack
