"""Test module for dicom.py"""

from srmip.dicom_to_nifti.dicom import Image


def test_metadata_is_unique():
    """Test if setting one Image's metadata does not touch another series."""
    new_image1 = Image()
    new_image2 = Image()

    assert new_image1.metadata == {}
    assert new_image2.metadata == {}

    new_image1._metadata["a"] = 0

    assert new_image1.metadata == {"a": 0}
    assert new_image2.metadata == {}
