"""Test module for dose.py"""


import logging
from pathlib import Path

from srmip.dose.dose import Dose
from srmip.image.image import Image

REFERENCE_DICOM_PATH = Path(__file__).parent / "Dicom" / "dicompyler_img"
REFERENCE_DICOM_IMAGE_PATH = REFERENCE_DICOM_PATH / "ct.0.dcm"
REFERENCE_DICOM_DOSE_PATH = REFERENCE_DICOM_PATH / "rtdose.dcm"
ORIGINAL_DOSE_SHAPE = (98, 129, 194)
ORIGINAL_DOSE_ORIGIN = (-228.6541915, -419.2444776, -122.4407)
ORIGINAL_DOSE_SPACING = (2.5, 2.5, 3.0)


def test_read_dose_without_reference(caplog):
    """Read DICOM RT Dose without reference image."""
    with caplog.at_level(logging.WARNING):
        dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH)
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "No reference image has been provided for the RT Dose." in caplog.text

    assert dose.numpy().shape == ORIGINAL_DOSE_SHAPE
    assert dose.origin == ORIGINAL_DOSE_ORIGIN
    assert dose.spacing == ORIGINAL_DOSE_SPACING
    assert len(dose.metadata) > 0


def test_read_dose_with_reference():
    """Read DICOM RT Dose with reference image."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    assert dose.numpy().shape == image.numpy().shape
    assert dose.origin == image.origin
    assert dose.spacing == image.spacing
    assert len(dose.metadata) > 0
