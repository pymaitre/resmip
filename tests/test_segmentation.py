"""Test module for segmentation objects."""

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from resmip.image import DicomModality, Image
from resmip.segmentation import Segmentation, SegmentationType

from .utils import (
    dicom_ct_path,
    dicom_rtst_path,
    generate_dummy_segmentation,
    ibsi_rtst_path,
)

TEST_SEG_SIZE = (204, 201, 60)
"""Size of the test Segmentation."""
TEST_SEG_SPACING = (0.97699999809265, 0.97699999809265, 2.9999999999998486)
"""Spacing of the test Segmentation."""


@pytest.mark.parametrize("threshold", [None, -1, 0, 0.5, 1, 5, 10, 11])
def test_generate_binary_segmentation(threshold):
    """Generate binary segmentation with various thresholds."""
    mask = np.linspace(0, 10, 300).reshape((3, 10, 10))
    dummy_segmentation = generate_dummy_segmentation(mask)
    assert dummy_segmentation.name == "dummy"
    np.testing.assert_allclose(dummy_segmentation.numpy(copy=False), mask)
    assert dummy_segmentation.segmentation_type == SegmentationType.fractional

    if threshold is None:
        binary_segmentation = dummy_segmentation.to_binary()
    else:
        if threshold <= 0:
            with pytest.raises(ValueError):
                _ = dummy_segmentation.to_binary(threshold=threshold)
            return
        binary_segmentation = dummy_segmentation.to_binary(threshold=threshold)
    assert isinstance(binary_segmentation, Segmentation)
    assert binary_segmentation.name == "dummy"
    assert binary_segmentation.origin == dummy_segmentation.origin
    assert binary_segmentation.direction == dummy_segmentation.direction
    assert binary_segmentation.spacing == dummy_segmentation.spacing
    if threshold == 11:
        assert np.all(binary_segmentation.numpy(copy=False) == 0)
    else:
        assert tuple(np.unique(binary_segmentation.numpy(copy=False))) == (0, 1)
    assert binary_segmentation.segmentation_type == SegmentationType.binary


@pytest.mark.parametrize("attr", ["origin", "spacing", "direction", "size"])
def test_segmentations_incompatible(attr):
    """Check segmentation compatibility."""
    mask = np.linspace(0, 10, 300).reshape((3, 10, 10))
    if attr in ["origin", "spacing"]:
        different_value = (1, 2, 1)
    elif attr == "direction":
        different_value = (0, 1, 0, 1, 0, 0, 0, 0, 1)
    elif attr == "size":
        pass
    else:
        raise NotImplementedError
    dummy_segmentation = generate_dummy_segmentation(mask)
    other_segmentation = generate_dummy_segmentation(mask)
    if attr != "size":
        setattr(other_segmentation, attr, different_value)  # pylint: disable=E0606
    else:
        other_segmentation = other_segmentation[10:, 5:]
    assert getattr(dummy_segmentation, attr) != getattr(other_segmentation, attr)
    assert dummy_segmentation.is_compatible(other_segmentation) is False
    assert other_segmentation.is_compatible(dummy_segmentation) is False


@pytest.mark.parametrize("use_structure_name", [True, False])
def test_read_single_nifti_segmentation(use_structure_name):
    """Read a nifti Segmentation with or without specifying a structure name."""
    segmentation_path = Path(__file__).parent / "Nifti" / "IBSI2_CT_phantom" / "mask" / "GTV-1.nii"
    structure_name = segmentation_path.stem
    if use_structure_name is True:
        structure_name = "structure"
        seg = Segmentation.read(segmentation_path, structure_name=structure_name)
    else:
        seg = Segmentation.read(segmentation_path)
    assert isinstance(seg, Segmentation)
    assert seg.name == structure_name
    # check size
    assert seg.size == TEST_SEG_SIZE
    # check spacing
    np.testing.assert_allclose(seg.spacing, TEST_SEG_SPACING)

    assert seg.modality == DicomModality.seg.value
    assert seg.segmentation_type == SegmentationType.binary


@pytest.mark.parametrize("use_structure_name", [True, False])
def test_read_single_fractional_nifti_segmentation(use_structure_name, tmp_path):
    """Read a fractional nifti Segmentation with or without specifying a structure name."""
    original_segmentation_path = (
        Path(__file__).parent / "Nifti" / "IBSI2_CT_phantom" / "mask" / "GTV-1.nii"
    )
    segmentation_path = tmp_path / original_segmentation_path.name
    original_seg = Segmentation.read(original_segmentation_path)
    original_seg[110, 70, 30] = 2
    original_seg.write(segmentation_path, write_metadata=False)
    structure_name = segmentation_path.stem
    if use_structure_name is True:
        structure_name = "structure"
        seg = Segmentation.read(segmentation_path, structure_name=structure_name)
    else:
        seg = Segmentation.read(segmentation_path)
    assert isinstance(seg, Segmentation)
    assert seg.name == structure_name
    # check size
    assert seg.size == TEST_SEG_SIZE
    # check spacing
    np.testing.assert_allclose(seg.spacing, TEST_SEG_SPACING)

    assert seg.modality == DicomModality.seg.value
    assert set(np.unique(seg.numpy(copy=False))) == set([0, 1, 2])
    assert seg.segmentation_type == SegmentationType.fractional


def test_read_single_dicom_structure_from_rtst(mock_dicom_image: Image):
    """Read a dicom RT Structure from file."""
    structure_name = "GTV-1"
    seg = Segmentation.read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )
    assert isinstance(seg, Segmentation)
    assert seg.segmentation_type == SegmentationType.binary
    assert seg.name == structure_name
    # check size
    assert seg.size == TEST_SEG_SIZE
    # check spacing
    np.testing.assert_allclose(seg.spacing, TEST_SEG_SPACING)


def test_read_single_dicom_structure_wrong_name(mock_dicom_image: Image):
    """Read a dicom RT Structure from file, with a wrong name."""
    structure_name = "gtv-1"
    with pytest.raises(KeyError):
        _ = Segmentation.read(
            dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
        )


def test_read_single_dicom_structure_without_reference_image():
    """Read a dicom RT Structure without specifying a reference image.

    A value error should be raised.
    """
    structure_name = "GTV-1"
    with pytest.raises(ValueError):
        _ = Segmentation.read(dicom_rtst_path(), structure_name=structure_name)


def test_read_single_dicom_structure_without_structure_name(mock_dicom_image: Image):
    """Read a dicom RT Structure without specifying a structure name.

    A value error should be raised.
    """
    with pytest.raises(ValueError):
        _ = Segmentation.read(dicom_rtst_path(), reference_image=mock_dicom_image)


def test_segmentation_resample():
    """Test Segmentation.resample()."""
    structure = Segmentation.read(ibsi_rtst_path())
    resampled_structure = structure.resample((0.8, 0.8, 0.8))
    volume = structure.numpy().sum() * np.prod(structure.spacing)
    resampled_volume = resampled_structure.numpy().sum() * np.prod(resampled_structure.spacing)
    assert np.allclose(resampled_volume, volume, rtol=0.006)


@pytest.mark.parametrize("left_shift", [-1, 0, 1])
@pytest.mark.parametrize("right_shift", [-1, 0, 1])
def test_segmentation_pad(left_shift, right_shift):
    """Test Segmentation.pad()."""
    structure_spacing = (1, 1, 1)
    structure_origin = np.array((0, 0, 0))
    reference_origin = structure_origin + left_shift
    structure_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    original_structure_shape = (5, 5, 5)
    reference_size = np.array(original_structure_shape) + right_shift
    original_array = np.zeros(original_structure_shape)
    point_coordinate = (2, 2, 2)
    original_array[point_coordinate] = 1
    original_structure = Segmentation.from_array(
        original_array,
        spacing=structure_spacing,
        origin=tuple(structure_origin.tolist()),
        direction=structure_direction,
        name="Struct",
    )
    reference_structure = Segmentation.from_array(
        np.zeros(reference_size),
        spacing=structure_spacing,
        origin=tuple(reference_origin.tolist()),
        direction=structure_direction,
        name="Struct",
    )
    padded_structure = original_structure.pad(reference_structure)
    new_coordinate = np.array(point_coordinate) - left_shift

    assert padded_structure.numpy().shape == reference_structure.numpy().shape
    assert padded_structure.numpy()[tuple(new_coordinate.tolist())] == 1
    assert padded_structure.name == reference_structure.name


def test_crop_structure(mock_dicom_structure: Segmentation):
    """Crop Segmentation."""
    mock_dicom_seg = Segmentation(mock_dicom_structure, name=mock_dicom_structure.name)
    cropped_seg = mock_dicom_seg[1:-2, 1:-2, 1:-2]
    padded_seg = cropped_seg.pad(mock_dicom_structure)

    np.testing.assert_equal(
        np.asarray(cropped_seg.GetSize()),
        np.asarray(mock_dicom_structure.GetSize()) - 3,
    )
    assert padded_seg.size == mock_dicom_structure.size
    assert padded_seg.origin == mock_dicom_structure.origin
    assert padded_seg.spacing == mock_dicom_structure.spacing
    np.testing.assert_equal(
        padded_seg.numpy(),
        mock_dicom_structure.numpy(),
    )


def test_segmentation_from_array(mock_dicom_segmentation: Segmentation):
    """Test Segmentation creation from a numpy array."""
    new_structure = Segmentation.from_array(
        mock_dicom_segmentation.numpy(),
        spacing=mock_dicom_segmentation.spacing,
        origin=mock_dicom_segmentation.origin,
        direction=mock_dicom_segmentation.direction,
        metadata=mock_dicom_segmentation.metadata,
        name=mock_dicom_segmentation.name,
    )
    assert new_structure.size == mock_dicom_segmentation.size
    assert new_structure.spacing == mock_dicom_segmentation.spacing
    assert new_structure.origin == mock_dicom_segmentation.origin
    assert new_structure.direction == mock_dicom_segmentation.direction
    assert np.all(new_structure.numpy() == mock_dicom_segmentation.numpy())
    assert new_structure.metadata == mock_dicom_segmentation.metadata
    assert new_structure.name == mock_dicom_segmentation.name


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_single_nifti_structure(extension, mock_dicom_structure: Segmentation, tmp_path):
    """Create a NIfTI Segmentation from reference."""
    rtst_path = tmp_path / f"{mock_dicom_structure.name}.{extension}"
    mock_dicom_structure.write(rtst_path)

    saved_seg = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(mock_dicom_structure.numpy(), sitk.GetArrayFromImage(saved_seg))


def test_write_dicom_segmentation_unsupported_modality(
    mock_dicom_segmentation: Segmentation, tmp_path
):
    """Save segmentation to unsupported dicom modality."""
    seg_path = tmp_path / "out.dcm"
    with pytest.raises(ValueError):
        mock_dicom_segmentation.write(filename=seg_path, modality="")


def test_written_dicom_segmentation_is_dicom_conformant(
    mock_dicom_segmentation: Segmentation, dicom_validator, tmp_path
):
    """Check that the written DICOM SEG follows DICOM standard."""
    output_path = tmp_path / "seg.dcm"
    mock_dicom_segmentation.write(output_path, reference_image_path=dicom_ct_path())

    result = next(iter(dicom_validator.validate(output_path).values()))

    assert result.errors == 0, result.module_errors
