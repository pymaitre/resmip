"""Common fixtures."""

# pylint: disable=W0621

import logging

import pydicom
import pytest
from dicom_validator.spec_reader.edition_reader import EditionReader
from dicom_validator.validator.dicom_file_validator import DicomFileValidator

import resmip

from .dicom_standard import DICOM_EDITION, DICOM_STANDARD_PATH
from .utils import dicom_ct_path, dicom_rtst_path, liver_dicom_seg


@pytest.fixture
def non_square_pixel_ct_path(tmp_path):
    """Copy of the liver CT series with 0.5 mm between rows and 0.8 mm between columns.

    PixelSpacing is row spacing first, SimpleITK spacing is x first. With square pixels the
    two orders coincide, so only non-square pixels expose a conversion that skips the swap.
    """
    source_path = liver_dicom_seg().parent / "liver_ct"
    series_path = tmp_path / "non_square_pixel_ct"
    series_path.mkdir()
    for slice_path in sorted(source_path.iterdir()):
        dataset = pydicom.dcmread(slice_path)
        dataset.PixelSpacing = [0.5, 0.8]
        dataset.save_as(series_path / slice_path.name)
    return series_path


@pytest.fixture
def mock_dicom_image():
    """Mock DICOM CT used in test modules."""
    return resmip.Image.read(dicom_ct_path())


@pytest.fixture
def mock_dicom_structure(mock_dicom_image: resmip.Image):
    """Mock DICOM structure (GTV-1) used in test modules."""
    structure_name = "GTV-1"
    return resmip.RTStructure.read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )


@pytest.fixture
def mock_dicom_segmentation(mock_dicom_image: resmip.Image):
    """Mock DICOM segmentation (GTV-1) used in test modules."""
    structure_name = "GTV-1"
    return resmip.Segmentation.read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )


@pytest.fixture
def dicom_validator():
    """Validator used to check if generated DICOM files are valid."""
    dicom_info = EditionReader(DICOM_STANDARD_PATH).dicom_info_for_edition(DICOM_EDITION)
    return DicomFileValidator(dicom_info, log_level=logging.WARNING)
