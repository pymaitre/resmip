"""Test module for constants.py"""


from pathlib import Path

import pydicom

from srmip.dicom_utils.constants import DICOM_FIELDS


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
    for dicom_name, dicom_tag_str in DICOM_FIELDS.items():
        dicom_tag = [hex(int(number, 16)) for number in dicom_tag_str.split("|")]
        assert header[dicom_name].value == header[dicom_tag].value
