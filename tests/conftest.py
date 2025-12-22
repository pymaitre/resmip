"""Common fixtures."""

import pytest

import resmip

from .utils import dicom_ct_path


@pytest.fixture
def mock_dicom_image():
    """Mock DICOM CT used in test modules."""
    return resmip.Image.read(dicom_ct_path())
