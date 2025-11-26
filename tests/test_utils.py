"""Test module for dicom.py"""

import pytest

from resmip.utils import format_digit_string


@pytest.mark.parametrize("is_digit", [True, False])
@pytest.mark.parametrize("trailing_whitespaces", [0, 1, 4])
@pytest.mark.parametrize("dots", [0, 1, 2])
@pytest.mark.parametrize("leading_minus", [0, 1, 2])
def test_format_digit_string(is_digit, trailing_whitespaces, dots, leading_minus):
    """Test function for format_digit_string."""
    indexes = [0, 1]
    if is_digit:
        test_strings = ["16576719", "16576719"]
    else:
        test_strings = ["165aop7671a9", "165aop7671a9"]
    for _ in range(trailing_whitespaces):
        for i in indexes:
            test_strings[i] += " "
            if is_digit and (dots in [0, 1] and leading_minus in [0, 1]):
                break
    for _ in range(dots):
        for i in indexes:
            test_strings[i] = test_strings[i][:4] + "." + test_strings[i][4:]
    for _ in range(leading_minus):
        for i in indexes:
            test_strings[i] = "-" + test_strings[i]

    assert format_digit_string(test_strings[0]) == test_strings[1]
