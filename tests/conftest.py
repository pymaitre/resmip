"""Common fixtures."""

# pylint: disable=W0621

import logging

import pytest
from dicom_validator.spec_reader.edition_reader import EditionReader
from dicom_validator.validator.dicom_file_validator import DicomFileValidator

import resmip

from .utils import dicom_ct_path, dicom_rtst_path

DICOM_EDITION = "2026c"


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
    dicom_info = EditionReader().dicom_info_for_edition(DICOM_EDITION)
    return DicomFileValidator(dicom_info, log_level=logging.WARNING)
