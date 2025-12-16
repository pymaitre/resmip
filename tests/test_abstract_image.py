"""Test module for Image class and it's children."""

import logging
from pathlib import Path

import numpy as np
import pytest

from resmip import Dose, Image, RTStructure
from resmip.image import CoregistrationMetric

from .utils import dicom_ct_path, ibsi_rtst_path


def mock_image():
    return Image.read_image(dicom_ct_path())


def mock_structure():
    return RTStructure.read_image(ibsi_rtst_path())


def mock_dose():
    REFERENCE_DICOM_PATH = Path(__file__).parent / "Dicom" / "dicompyler_img"
    REFERENCE_DICOM_IMAGE_PATH = REFERENCE_DICOM_PATH / "ct.0.dcm"
    REFERENCE_DICOM_DOSE_PATH = REFERENCE_DICOM_PATH / "rtdose.dcm"
    image = Image.read_image(REFERENCE_DICOM_IMAGE_PATH)
    return Dose.read_image(REFERENCE_DICOM_DOSE_PATH, reference_image=image)


def assert_object_compatible(obj1, obj2, obj_type=None):
    assert isinstance(obj1, type(obj2))
    assert isinstance(obj2, type(obj1))
    if obj_type:
        assert isinstance(obj1, obj_type)
        assert isinstance(obj2, obj_type)
    assert len(obj1.metadata) == len(obj2.metadata)
    if isinstance(obj1, RTStructure):
        assert obj1.name == obj2.name


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_sum(image, factor, caplog):
    """Add constant factor to image pixels."""
    input_image: Image = image()
    image_type = type(input_image)

    if isinstance(input_image, RTStructure):
        with pytest.raises(NotImplementedError):
            _ = input_image + factor
        return
    if factor < 0:
        with caplog.at_level(logging.WARNING):
            _ = input_image + factor
        for record in caplog.records:
            assert record.levelname == "WARNING"
        input_image = input_image.astype(int)

    assert_object_compatible(input_image, input_image, obj_type=image_type)
    summed_image = input_image + factor
    assert_object_compatible(input_image, summed_image, obj_type=image_type)
    assert np.all(summed_image.numpy() == input_image.numpy() + factor)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_subtract(image, factor, caplog):
    """Remove constant factor to dose pixels."""
    input_image: Image = image()
    image_type = type(input_image)
    if isinstance(input_image, RTStructure):
        with pytest.raises(NotImplementedError):
            _ = input_image + factor
        return
    with caplog.at_level(logging.WARNING):
        _ = input_image - factor
    for record in caplog.records:
        assert record.levelname == "WARNING"
    input_image = input_image.astype(int)
    assert_object_compatible(input_image, input_image, obj_type=image_type)
    subtracted_image = input_image - factor
    assert_object_compatible(input_image, subtracted_image, obj_type=image_type)
    assert np.all(subtracted_image.numpy() == input_image.numpy() - factor)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_multiply(image, factor, caplog):
    """Multiply constant factor to image pixels."""
    input_image: Image = image()
    image_type = type(input_image)
    if isinstance(input_image, RTStructure):
        with pytest.raises(NotImplementedError):
            _ = input_image + factor
        return
    if factor < 0:
        with caplog.at_level(logging.WARNING):
            _ = input_image * factor
        for record in caplog.records:
            assert record.levelname == "WARNING"
        input_image = input_image.astype(int)
    assert_object_compatible(input_image, input_image, obj_type=image_type)
    multiplied_image = input_image * factor
    assert_object_compatible(input_image, multiplied_image, obj_type=image_type)
    assert np.all(multiplied_image.numpy() == input_image.numpy() * factor)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_divide(image, factor):
    """Divide constant factor to image pixels."""
    input_image: Image = image()
    image_type = type(input_image)
    if isinstance(input_image, RTStructure):
        with pytest.raises(NotImplementedError):
            _ = input_image + factor
        return
    divided_image = input_image / factor
    assert_object_compatible(input_image, divided_image, obj_type=image_type)
    assert np.all(divided_image.numpy() == input_image.numpy() / factor)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("dtype", [int, np.float32, np.uint32])
def test_image_astype(image, dtype):
    """Test image type casting."""
    input_image: Image = image()
    image_type = type(input_image)
    cast_image = input_image.astype(dtype)
    assert_object_compatible(cast_image, input_image, image_type)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
@pytest.mark.parametrize("dtype", [int, np.float32, np.uint32])
def test_image_getitem(image, dtype):
    """Test image getitem (for slicing/cropping)."""
    input_image: Image = image()
    image_type = type(input_image)
    cast_image = input_image[10, :5]
    assert_object_compatible(cast_image, input_image, image_type)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
def test_image_from_array(image):
    """Test image getitem (for slicing/cropping)."""
    input_image: Image = image()
    image_type = type(input_image)
    extra_args = {}
    if isinstance(input_image, RTStructure):
        extra_args["name"] = input_image.name
    new_image = image_type.from_array(
        input_image.numpy(),
        spacing=input_image.spacing,
        origin=input_image.origin,
        direction=input_image.direction,
        metadata=input_image.metadata,
        **extra_args,
    )
    assert_object_compatible(new_image, input_image, image_type)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
def test_image_resample(image):
    """Test image resampling."""
    input_image: Image = image()
    image_type = type(input_image)
    scale = 0.5
    new_spacing = np.array(input_image.spacing) / scale
    resampled_image = input_image.resample(new_spacing.tolist())
    assert_object_compatible(resampled_image, input_image, image_type)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
def test_image_pad(image):
    """Test image padding."""
    input_image: Image = image()
    image_type = type(input_image)
    larger_shape = (20, 20, 20)
    smaller_shape = (18, 18, 18)
    image_spacing = (1, 1, 1)
    image_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    extra_args = {}
    if isinstance(input_image, RTStructure):
        extra_args["name"] = input_image.name
    larger_image = image_type.from_array(
        np.ones(larger_shape), spacing=image_spacing, origin=(0, 0, 0), direction=image_direction
    )
    smaller_image = image_type.from_array(
        np.ones(smaller_shape),
        spacing=image_spacing,
        origin=(1, 1, 1),
        direction=image_direction,
        metadata=input_image.metadata,
        **extra_args,
    )
    padded_image = smaller_image.pad(larger_image)
    assert_object_compatible(padded_image, input_image, image_type)


@pytest.mark.parametrize("image", [mock_image, mock_structure, mock_dose])
def test_image_coregistration(image):
    """Coregister images."""
    input_image: Image = image()
    reference_image: Image = image()
    image_type = type(input_image)
    if image_type == Dose:  # mock dose does not have enough pixels on the z axis
        input_image = Dose(mock_image())
        reference_image = Dose(mock_image())
    assert input_image.origin == reference_image.origin
    input_image.origin = (0, 0, 0)
    assert input_image.origin != reference_image.origin
    coregistration_args = dict(
        reference_image=reference_image,
        fill_value=-1000,
        coregistration_metric=CoregistrationMetric.correlation,
        seed=1,
        num_threads=1,
    )
    if isinstance(input_image, (RTStructure, Dose)):
        with pytest.raises(NotImplementedError):
            _ = input_image.coregister(
                **coregistration_args,
            )
        return
    coregistered_image = input_image.coregister(
        **coregistration_args,
    )
    assert_object_compatible(coregistered_image, input_image, image_type)
