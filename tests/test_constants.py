"""Test module for constants.py."""

from pathlib import Path

import pydicom
import pytest
from pydicom.tag import Tag

from resmip.dicom_utils.constants import (
    DICOM_FIELDS,
    dicom_tag_to_string,
    string_tag_for_keyword,
)


def dicom_ct_path() -> Path:
    """Path of the test Dicom CT."""
    return Path(__file__).parent / "Dicom" / "siemens_mprage_0_dcm"


def test_name_tag_correspond():
    """Test that name and tag in the Dicom header correspond to the same value."""
    # pick only one dicom file
    dicom_file_path = list(dicom_ct_path().glob("*.dcm"))[0]
    header = pydicom.dcmread(dicom_file_path)
    # Add missing header values
    header.add_new(0x00180088, "DS", header["SliceThickness"].value)
    for dicom_name in DICOM_FIELDS:
        dicom_tag_str = string_tag_for_keyword(dicom_name)
        dicom_tag = [hex(int(number, 16)) for number in dicom_tag_str.split("|")]
        assert header[dicom_name].value == header[dicom_tag].value


@pytest.mark.parametrize(
    "tag",
    [
        {"int": Tag(0x12345678), "str": "1234|5678"},
        {"int": Tag(0x345678), "str": "0034|5678"},
        {"int": Tag(0x340078), "str": "0034|0078"},
    ],
)
def test_dicom_tag_to_string(tag):
    """Test conversion of DICOM tags into strings."""
    expected_tag = tag["str"]
    generated_tag = dicom_tag_to_string(tag["int"])
    assert generated_tag == expected_tag
