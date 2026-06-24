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
        dose = Dose.read(REFERENCE_DICOM_DOSE_PATH)
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
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
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

    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    with pytest.raises(KeyError):
        Dose.read(modified_dicom_dose_path, reference_image=image)


def test_write_dose_nifti(tmp_path):
    """Write dose to nifti file."""
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / "dose.nii.gz"
    dose.write(nifti_dose_path)

    nifti_dose = Dose.read(nifti_dose_path)
    assert np.all(nifti_dose.numpy() == dose.numpy())


@pytest.mark.parametrize("read_existing_dose", [True, False])
def test_dose_modality(read_existing_dose):
    """Check DICOM modality of ``Dose``."""
    if read_existing_dose:
        image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
        dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    else:
        dose = Dose()
    assert isinstance(dose, Dose)
    assert dose.modality == "RTDOSE"


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_should_not_write_metadata_when_write_metadata_false_dir(extension, tmp_path):
    """Sidecar JSON absent when write_metadata=False, directory mode."""
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / f"dose.{extension}"
    metadata_path = nifti_dose_path.parent / f".{nifti_dose_path.stem}.json"
    dose.write(nifti_dose_path, write_metadata=False)
    assert not metadata_path.exists()


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_should_write_metadata_by_default(extension, tmp_path):
    """Sidecar JSON present with the default write_metadata=True."""
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / f"dose.{extension}"
    metadata_path = nifti_dose_path.parent / f".{nifti_dose_path.stem}.json"
    dose.write(nifti_dose_path)
    assert metadata_path.exists()


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_should_not_read_metadata_when_read_metadata_false(extension, tmp_path):
    """Sidecar-only metadata absent when reading with read_metadata=False."""
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / f"dose.{extension}"
    sentinel_key = "resmip_test_key"
    dose.metadata[sentinel_key] = "resmip_test_value"
    dose.write(nifti_dose_path, write_metadata=True)
    loaded = Dose.read(nifti_dose_path, read_metadata=False)
    assert sentinel_key not in loaded.metadata


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_should_read_metadata_by_default(extension, tmp_path):
    """Sidecar metadata present with default read_metadata=True."""
    image = Image.read(REFERENCE_DICOM_IMAGE_PATH)
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, reference_image=image)
    nifti_dose_path = tmp_path / f"dose.{extension}"
    sentinel_key = "resmip_test_key"
    dose.metadata[sentinel_key] = "resmip_test_value"
    dose.write(nifti_dose_path, write_metadata=True)
    loaded = Dose.read(nifti_dose_path)
    assert loaded.metadata[sentinel_key] == "resmip_test_value"
