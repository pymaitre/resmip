"""Test module for rt_utils_wrapper.py."""

import logging

import numpy as np
import pydicom
import pytest

import srmip
from srmip.dicom_utils.rt_utils_wrapper import RTStruct, get_slice_positioning

from .utils import dicom_ct_path, dicom_rtst_path


def test_validate_mask_array_wrong_type():
    """Test validate_mask_array with wrong array type."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    mask = np.zeros(image.GetSize(), dtype=np.uint8)
    with pytest.raises(TypeError):
        rtst.validate_mask_array(mask)


def test_validate_mask_array_wrong_shape():
    """Test validate_mask_array with wrong array shape."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    wrong_shape = np.array(image.GetSize()) - 1
    mask = np.zeros(wrong_shape, dtype=bool)
    with pytest.raises(ValueError):
        rtst.validate_mask_array(mask)


def test_validate_mask_array_wrong_dimension():
    """Test validate_mask_array with wrong dimension (2D array)."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    wrong_shape = image.GetSize()[:2]
    mask = np.zeros(wrong_shape, dtype=bool)
    with pytest.raises(ValueError):
        rtst.validate_mask_array(mask)


def test_validate_mask_array_empty(caplog):
    """Test validate_mask_array with all zeros."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    mask = np.zeros(image.GetSize(), dtype=bool)
    with caplog.at_level(logging.INFO):
        rtst.validate_mask_array(mask)
    for record in caplog.records:
        assert record.levelname == "INFO"
    assert "ROI mask is empty" in caplog.text


def test_validate_mask_wrong_spacing():
    """Test validate_mask_array with wrong voxel spacing."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    structure = srmip.RTStructure.read_image(
        dicom_rtst_path(), structure_name="GTV-1", reference_image=image
    ).resample((0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        rtst.validate_mask(structure)


def test_validate_mask_wrong_origin():
    """Test validate_mask_array with wrong origin."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    structure = srmip.RTStructure.read_image(
        dicom_rtst_path(), structure_name="GTV-1", reference_image=image
    )
    structure.origin = (0, 0, 0)
    with pytest.raises(ValueError):
        rtst.validate_mask(structure)


def test_validate_mask_wrong_direction():
    """Test validate_mask_array with wrong orientation."""
    rtst = RTStruct.create_new(dicom_ct_path())
    image = srmip.Image.read_image(dicom_ct_path())
    structure = srmip.RTStructure.read_image(
        dicom_rtst_path(), structure_name="GTV-1", reference_image=image
    )
    structure.direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError):
        rtst.validate_mask(structure)


@pytest.mark.parametrize(
    "key",
    [
        "SOPInstanceUID",
        "SliceThickness",
        "PixelSpacing",
        "ImagePositionPatient",
        "ImageOrientationPatient",
    ],
)
def test_get_slice_positioning_missing_key(key):
    """Test validate_mask_array with missing keys in the header."""
    dicom_file = list(dicom_ct_path().glob("*.dcm"))[0]
    dicom_slice = pydicom.dcmread(dicom_file)
    del dicom_slice[key]
    with pytest.raises(KeyError):
        get_slice_positioning(dicom_slice)
