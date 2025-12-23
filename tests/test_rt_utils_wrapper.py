"""Test module for rt_utils_wrapper.py."""

# pylint: disable=W0621

import json
import logging

import numpy as np
import pydicom
import pytest

import resmip
from resmip.dicom_utils.rt_utils_wrapper import RTStruct
from resmip.dicom_utils.rt_utils_wrapper.header import (
    add_leading_zero_to_header_value,
    get_slice_positioning,
)
from resmip.dicom_utils.rt_utils_wrapper.roidata import ROIData

from .utils import dicom_ct_path


def test_validate_mask_array_wrong_type(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure
):
    """Test validate_mask_array with wrong array type."""
    rtst = RTStruct.create_new(dicom_ct_path())
    roidata = ROIData(mock_dicom_structure, 1, mock_dicom_structure.name, "0")
    mask = np.zeros(mock_dicom_image.GetSize(), dtype=np.uint8)
    with pytest.raises(TypeError):
        roidata.validate_mask_array(mask, rtst.series_data)


def test_validate_mask_array_wrong_shape(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure
):
    """Test validate_mask_array with wrong array shape."""
    rtst = RTStruct.create_new(dicom_ct_path())
    roidata = ROIData(mock_dicom_structure, 1, mock_dicom_structure.name, "0")
    wrong_shape = np.array(mock_dicom_image.GetSize()) - 1
    mask = np.zeros(wrong_shape, dtype=bool)
    with pytest.raises(ValueError):
        roidata.validate_mask_array(mask, rtst.series_data)


def test_validate_mask_array_wrong_dimension(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure
):
    """Test validate_mask_array with wrong dimension (2D array)."""
    rtst = RTStruct.create_new(dicom_ct_path())
    roidata = ROIData(mock_dicom_structure, 1, mock_dicom_structure.name, "0")
    wrong_shape = mock_dicom_image.GetSize()[:2]
    mask = np.zeros(wrong_shape, dtype=bool)
    with pytest.raises(ValueError):
        roidata.validate_mask_array(mask, rtst.series_data)


def test_validate_mask_array_empty(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure, caplog
):
    """Test validate_mask_array with all zeros."""
    rtst = RTStruct.create_new(dicom_ct_path())
    roidata = ROIData(mock_dicom_structure, 1, mock_dicom_structure.name, "0")
    mask = np.zeros(mock_dicom_image.GetSize(), dtype=bool)
    with caplog.at_level(logging.INFO):
        roidata.validate_mask_array(mask, rtst.series_data)
    for record in caplog.records:
        assert record.levelname == "INFO"
    assert "ROI mask is empty" in caplog.text


def test_validate_mask_wrong_spacing(mock_dicom_structure: resmip.RTStructure):
    """Test validate_mask_array with wrong voxel spacing."""
    rtst = RTStruct.create_new(dicom_ct_path())
    structure = mock_dicom_structure.resample((0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        rtst.validate_mask(structure)


def test_validate_mask_wrong_origin(mock_dicom_structure: resmip.RTStructure):
    """Test validate_mask_array with wrong origin."""
    rtst = RTStruct.create_new(dicom_ct_path())
    mock_dicom_structure.origin = (0, 0, 0)
    with pytest.raises(ValueError):
        rtst.validate_mask(mock_dicom_structure)


def test_validate_mask_wrong_direction(mock_dicom_structure: resmip.RTStructure):
    """Test validate_mask_array with wrong orientation."""
    rtst = RTStruct.create_new(dicom_ct_path())
    mock_dicom_structure.direction = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    with pytest.raises(ValueError):
        rtst.validate_mask(mock_dicom_structure)


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


@pytest.mark.parametrize(
    "spacing_value_1",
    ["0.9790265", "1.9790265", ".9790265", "10.9790265"],
)
@pytest.mark.parametrize(
    "spacing_value_2",
    ["0.9790265", "1.9790265", ".9790265", "10.9790265"],
)
def test_add_leading_zero_to_header_value(spacing_value_1, spacing_value_2):
    """Test the function for adding leading zeros to pixel spacing."""
    old_spacing = f"[{spacing_value_1}, {spacing_value_2}]"
    old_spacing_f = f"[{float(spacing_value_1)}, {float(spacing_value_2)}]"
    new_spacing = add_leading_zero_to_header_value(old_spacing)
    assert str([float(value) for value in json.loads(new_spacing)]) == old_spacing_f
