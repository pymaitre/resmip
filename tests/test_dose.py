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
IDENTITY_DIRECTION = (1, 0, 0, 0, 1, 0, 0, 0, 1)
X_FLIPPED_DIRECTION = (-1, 0, 0, 0, 1, 0, 0, 0, 1)
BASE_SIZE_XYZ = (40, 40, 20)
BASE_SPACING_XYZ = (1.0, 1.0, 2.0)
RAMP_DOSE_SIZE_XYZ = (10, 10, 8)
RAMP_DOSE_SPACING_XYZ = (2.5, 2.5, 3.0)
INTERIOR_MARGIN_VOXELS = 1.5
RAMP_TOLERANCE = 1e-3


def _base_image(origin_xyz: tuple, direction: tuple) -> Image:
    """Build a CT-like reference grid whose spacing differs from the ramp dose's."""
    return Image.from_array(
        np.zeros(BASE_SIZE_XYZ[::-1], dtype=np.float32),
        spacing=BASE_SPACING_XYZ,
        origin=origin_xyz,
        direction=direction,
    )


def _write_ramp_dose(dose_path: Path, origin_xyz: tuple, direction: tuple) -> Dose:
    """Write a dose whose value at each voxel centre is that centre's physical x.

    Linear interpolation reproduces a linear ramp exactly, so after alignment the
    difference between a voxel's value and its own physical x measures how far the
    content was displaced, free of interpolation blur.
    """
    grid = _ramp_grid(np.zeros(RAMP_DOSE_SIZE_XYZ[::-1], dtype=np.float32), origin_xyz, direction)
    ramp_zyx = np.zeros(RAMP_DOSE_SIZE_XYZ[::-1], dtype=np.float32)
    for index_zyx in np.ndindex(ramp_zyx.shape):
        index_xyz = [int(index) for index in reversed(index_zyx)]
        ramp_zyx[index_zyx] = grid.TransformIndexToPhysicalPoint(index_xyz)[0]
    ramp_dose = Dose(_ramp_grid(ramp_zyx, origin_xyz, direction))
    ramp_dose.write(dose_path)
    return ramp_dose


def _ramp_grid(array_zyx: np.ndarray, origin_xyz: tuple, direction: tuple) -> Image:
    """Place an array on the ramp dose's grid."""
    return Image.from_array(
        array_zyx, spacing=RAMP_DOSE_SPACING_XYZ, origin=origin_xyz, direction=direction
    )


def _max_ramp_error_inside(aligned_dose: Image, ramp_dose: Image) -> float:
    """Return the largest |value - physical x| over aligned voxels well inside the ramp dose.

    Voxels within ``INTERIOR_MARGIN_VOXELS`` of the ramp dose's boundary are excluded:
    there, interpolation correctly mixes the ramp with the zero fill outside the dose,
    which is not displacement.
    """
    upper_limit_xyz = np.array(ramp_dose.size) - 1 - INTERIOR_MARGIN_VOXELS
    aligned_zyx = aligned_dose.numpy(copy=False)
    errors = []
    for index_zyx in np.ndindex(aligned_zyx.shape):
        index_xyz = [int(index) for index in reversed(index_zyx)]
        point_xyz = aligned_dose.TransformIndexToPhysicalPoint(index_xyz)
        continuous_index_xyz = np.array(
            ramp_dose.TransformPhysicalPointToContinuousIndex(point_xyz)
        )
        if np.all(continuous_index_xyz >= INTERIOR_MARGIN_VOXELS) and np.all(
            continuous_index_xyz <= upper_limit_xyz
        ):
            errors.append(abs(float(aligned_zyx[index_zyx]) - point_xyz[0]))
    return max(errors)


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


def test_scale_dose_values_by_dose_grid_scaling_when_no_reference_image():
    """Doses read without a reference image are still scaled to physical units."""
    header = pydicom.dcmread(REFERENCE_DICOM_DOSE_PATH)
    expected_dose_zyx = header.pixel_array * float(header.DoseGridScaling)

    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH)

    assert np.allclose(dose.numpy(copy=False), expected_dose_zyx, rtol=1e-9, atol=0)


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


def test_dose_keeps_physical_position_when_origin_off_reference_grid(tmp_path):
    """A dose whose origin is off the reference lattice is aligned without displacement."""
    dose_path = tmp_path / "dose.nii.gz"
    ramp_dose = _write_ramp_dose(dose_path, (3.3, 4.6, 5.0), IDENTITY_DIRECTION)
    base_image = _base_image((0.0, 0.0, 0.0), IDENTITY_DIRECTION)

    aligned_dose = Dose.read(dose_path, reference_image=base_image)

    assert _max_ramp_error_inside(aligned_dose, ramp_dose) < RAMP_TOLERANCE


def test_dose_keeps_physical_position_when_directions_share_x_flip(tmp_path):
    """A dose sharing a non-identity direction with the reference is aligned in place."""
    dose_path = tmp_path / "dose.nii.gz"
    ramp_dose = _write_ramp_dose(dose_path, (35.7, 4.6, 5.0), X_FLIPPED_DIRECTION)
    base_image = _base_image((39.0, 0.0, 0.0), X_FLIPPED_DIRECTION)

    aligned_dose = Dose.read(dose_path, reference_image=base_image)

    assert _max_ramp_error_inside(aligned_dose, ramp_dose) < RAMP_TOLERANCE


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


def test_read_accepts_rtdose_modality():
    """A dose read with modality RTDOSE is an RTDOSE."""
    dose = Dose.read(REFERENCE_DICOM_DOSE_PATH, modality="RTDOSE")

    assert dose.modality == "RTDOSE"


def test_read_raises_when_modality_is_not_rtdose():
    """A dose cannot be read as any modality other than RTDOSE."""
    with pytest.raises(ValueError, match="CT"):
        Dose.read(REFERENCE_DICOM_DOSE_PATH, modality="CT")
