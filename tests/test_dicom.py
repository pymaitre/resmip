"""Test module for dicom.py"""

from pathlib import Path

from srmip.dicom_to_nifti.dicom import Image, read_dicom_series


def dicom_ct_path() -> Path:
    """Path of the test Dicom CT."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "image"


def test_metadata_is_unique():
    """Test if setting one Image's metadata does not touch another series."""
    new_image1 = Image()
    new_image2 = Image()

    assert new_image1.metadata == {}
    assert new_image2.metadata == {}

    new_image1.metadata["a"] = 0

    assert new_image1.metadata == {"a": 0}
    assert new_image2.metadata == {}


def test_metadata_contains_only_strings():
    """Check that all elements in the read Dicom header are python strings."""
    dicom_image = read_dicom_series(dicom_ct_path())

    for element in dicom_image.metadata.values():
        assert isinstance(element, str)
