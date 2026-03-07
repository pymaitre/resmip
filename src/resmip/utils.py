"""Functions any data types used globally."""

import os
import re
from typing import Union

__all__ = ["format_dicom_code_string", "format_digit_string", "PathLike"]

PathLike = Union[str, os.PathLike]
"""Types used in classes and functions for file names."""


def format_digit_string(digit_string: str) -> str:
    """Reformat a string containing a digit.

    Remove trailing whitespaces. If the input string does not represent
    a string, do nothing.

    :param digit_string: python string containing the digit.
    :type digit_string: str
    :return: python string containing the digit without trailing whitespaces.
    :rtype: str
    """
    captured_string = re.sub(
        r"^-?(\d)*([0-9]\.|\.[0-9])?(\d)* *$", r"\g<1>\g<3>\g<2>", digit_string
    )
    if re.sub(r"\.", "", captured_string).isdigit():
        digit_string = re.sub(" *$", "", digit_string)
    return digit_string


def format_dicom_code_string(code_string: str) -> str:
    """Strip the string of unnecessary characters.

    More information here:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
    """
    return code_string.strip()
