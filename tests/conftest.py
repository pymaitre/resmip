"""Common fixtures."""

# pylint: disable=W0621

import pytest

import resmip
from resmip.segmentation import Segmentation

from .utils import dicom_ct_path, dicom_rtst_path


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
    return Segmentation.read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )
