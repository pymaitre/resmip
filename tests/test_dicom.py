"""Test module for dicom.py"""

import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

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


@pytest.mark.parametrize("is_string", [True, False])
def test_metadata_file_name(is_string, tmp_path):
    """Test that the metadata file name has the correct name."""
    nifti_file_name = tmp_path / "nifti_image.nii"
    if is_string:
        given_nifti_file_name = str(nifti_file_name)
    else:
        given_nifti_file_name = nifti_file_name
    metadata_file_name = Image().metadata_file_name(given_nifti_file_name)
    assert metadata_file_name == tmp_path / f".{nifti_file_name.stem}.json"


def test_dicom_image_pixel_array():
    image = read_dicom_series(dicom_ct_path())
    image_array = sitk.GetArrayFromImage(image)
    min_instance_number = np.array(json.loads(image.metadata["slice_indexes"])).max()
    for dicom_file in dicom_ct_path().glob("*.dcm"):
        dataset = pydicom.dcmread(dicom_file)
        if dataset["Modality"].value != "CT":
            continue
        slice_index = min_instance_number - dataset["InstanceNumber"].value
        pixel_array = (
            np.frombuffer(dataset.PixelData, dtype=np.int16).reshape(
                (dataset["Rows"].value, dataset["Columns"].value)
            )
            - 1000
        )
        assert np.all(pixel_array == image_array[slice_index, :, :])
