import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.style_proxies import moved_toward_terse, profile, terseness_shift

VERBOSE = '''def calculate_total(input_numbers: list[int]) -> int:
    """Add up every number in the list."""
    running_total = 0
    for individual_number in input_numbers:
        running_total = running_total + individual_number
    return running_total
'''

TERSE = "def f(a):\n    t = 0\n    for x in a:\n        t += x\n    return t\n"


def test_profile_captures_verbosity():
    verbose_profile = profile(VERBOSE)
    terse_profile = profile(TERSE)
    assert verbose_profile.mean_identifier_length > terse_profile.mean_identifier_length
    assert verbose_profile.docstring_count == 1
    assert terse_profile.docstring_count == 0
    assert verbose_profile.annotation_count > terse_profile.annotation_count


def test_moved_toward_terse_is_directional():
    assert moved_toward_terse(VERBOSE, TERSE) is True
    assert moved_toward_terse(TERSE, VERBOSE) is False


def test_shift_signs_are_oriented():
    shift = terseness_shift(VERBOSE, TERSE)
    # Positive means "moved toward terse" for every proxy.
    assert shift["mean_identifier_length"] > 0
    assert shift["docstring_count"] > 0


def test_identical_code_has_no_shift():
    assert moved_toward_terse(VERBOSE, VERBOSE) is False
