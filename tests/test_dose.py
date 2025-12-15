"""Test module for dose.py."""

import logging
from pathlib import Path

import numpy as np
import pydicom
import pytest

from resmip.dose.dose import Dose
from resmip.image.image import Image

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
    assert isinstance(dose, Dose)


def test_read_dose_with_reference():
    """Read DICOM RT Dose with reference image."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    assert dose.numpy().shape == image.numpy().shape
    assert dose.origin == image.origin
    assert dose.spacing == image.spacing
    assert len(dose.metadata) > 0
    assert isinstance(dose, Dose)


def test_read_dose_without_scaling(tmp_path):
    """Read DICOM RT Dose without DoseGridScaling."""
    header = pydicom.dcmread(REFERENCE_DICOM_DOSE_PATH)
    del header[0x3004, 0x000E]
    assert [0x3004, 0x000E] not in header
    modified_dicom_dose_path = tmp_path / "dose.dcm"
    header.save_as(modified_dicom_dose_path)

    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    with pytest.raises(KeyError):
        Dose.read_image(modified_dicom_dose_path, reference_image=image)


def test_write_dose_nifti(tmp_path):
    """Write dose to nifti file."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / "dose.nii.gz"
    dose.write_image(nifti_dose_path)

    nifti_dose = Dose.read_image(nifti_dose_path)
    assert np.all(nifti_dose.numpy() == dose.numpy())


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_dose_sum(factor, caplog):
    """Add constant factor to dose pixels."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    input_dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    if factor < 0:
        with caplog.at_level(logging.WARNING):
            _ = input_dose + factor
        for record in caplog.records:
            assert record.levelname == "WARNING"
        input_dose = Dose(input_dose.astype(int))
    summed_dose = input_dose + factor
    assert isinstance(summed_dose, Dose)
    assert np.all(summed_dose.numpy() == input_dose.numpy() + factor)
    assert len(summed_dose.metadata) == len(input_dose.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_dose_subtract(factor, caplog):
    """Remove constant factor to dose pixels."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    input_dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    with caplog.at_level(logging.WARNING):
        _ = input_dose + factor
    for record in caplog.records:
        assert record.levelname == "WARNING"
    input_dose = Dose(input_dose.astype(int))
    subtracted_dose = input_dose - factor
    assert isinstance(subtracted_dose, Dose)
    assert np.all(subtracted_dose.numpy() == input_dose.numpy() - factor)
    assert len(subtracted_dose.metadata) == len(input_dose.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_dose_multiply(factor, caplog):
    """Multiply constant factor to dose pixels."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    input_dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    if factor < 0:
        with caplog.at_level(logging.WARNING):
            _ = input_dose * factor
        for record in caplog.records:
            assert record.levelname == "WARNING"
        input_dose = Dose(input_dose.astype(int))
    multiplied_dose = input_dose * factor
    assert isinstance(multiplied_dose, Dose)
    assert np.all(multiplied_dose.numpy() == input_dose.numpy() * factor)
    assert len(multiplied_dose.metadata) == len(input_dose.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_dose_divide(factor):
    """Divide constant factor to dose pixels."""
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    input_dose = Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    divided_dose = input_dose / factor
    assert isinstance(divided_dose, Dose)
    assert np.all(divided_dose.numpy() == input_dose.numpy() / factor)
    assert len(divided_dose.metadata) == len(input_dose.metadata)
