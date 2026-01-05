"""Test module for image.py."""

# pylint: disable=W0621

import logging
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from pydicom import dcmread

import resmip.dicom_utils.series as dicom_series
from resmip.dicom_utils.constants import string_tag_for_keyword
from resmip.image import CoregistrationMetric, Image
from resmip.image.data_types import sitk_image_dtype
from resmip.image.image import _metadata_file_name
from resmip.image.metadata import SERIES_MODALITIES, DicomModality
from resmip.utils import format_digit_string

from .utils import coregistered_image_path, dicom_ct_path

REQUIRED_IMAGE_FIELDS = [
    string_tag_for_keyword(x)
    for x in [
        "Modality",
        "PatientID",
        "StudyInstanceUID",
    ]
]
"""DICOM fields that must be present in image metadata."""


def test_metadata_is_unique():
    """Test if setting one Image's metadata does not touch another series."""
    image_modality = "CT"
    new_image1 = Image(modality=image_modality)
    new_image2 = Image(modality=image_modality)

    metadata_1 = {elem: "" for elem in REQUIRED_IMAGE_FIELDS}
    metadata_2 = {elem: "" for elem in REQUIRED_IMAGE_FIELDS}
    metadata_1[string_tag_for_keyword("Modality")] = image_modality
    metadata_2[string_tag_for_keyword("Modality")] = image_modality
    for field in ["PatientID"]:
        metadata_1[string_tag_for_keyword(field)] = ""
        metadata_2[string_tag_for_keyword(field)] = ""
    assert new_image1.metadata == metadata_1
    assert new_image2.metadata == metadata_2

    new_image1.metadata["a"] = 0
    metadata_1["a"] = 0

    assert new_image1.metadata == metadata_1
    assert new_image2.metadata == metadata_2


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_metadata_contains_only_strings(file_format, mock_dicom_image: Image, tmp_path):
    """Check that all elements in the read Dicom header are python strings."""
    if file_format == "nifti":
        nifti_image_path = tmp_path / "image.nii"
        mock_dicom_image.write(nifti_image_path)
        dicom_image = Image().read(nifti_image_path)
    else:
        dicom_image = mock_dicom_image

    for element in dicom_image.metadata.values():
        assert isinstance(element, str)


@pytest.mark.parametrize("is_string", [True, False])
def test_metadata_file_name(is_string, tmp_path):
    """Test that the metadata file name has the correct name."""
    nifti_file_name = tmp_path / "nifti_image.nii"
    if is_string:
        given_nifti_file_name = str(nifti_file_name)
    else:
        given_nifti_file_name = nifti_file_name
    metadata_file_name = _metadata_file_name(given_nifti_file_name)
    assert metadata_file_name == tmp_path / f".{nifti_file_name.stem}.json"


def test_image_spacing_getter(mock_dicom_image: Image):
    """Test Image.spacing()."""
    assert mock_dicom_image.spacing == mock_dicom_image.GetSpacing()


def test_image_spacing_setter(mock_dicom_image: Image):
    """Test Image.spacing = value."""
    x_spacing = 0.5
    y_spacing = 1
    z_spacing = 5.2
    new_spacing = (x_spacing, y_spacing, z_spacing)
    mock_dicom_image.spacing = new_spacing
    assert mock_dicom_image.GetSpacing() == new_spacing
    assert mock_dicom_image.spacing == mock_dicom_image.GetSpacing()


def test_image_origin_getter(mock_dicom_image: Image):
    """Test Image.origin()."""
    assert mock_dicom_image.origin == mock_dicom_image.GetOrigin()


def test_image_origin_setter(mock_dicom_image: Image):
    """Test Image.origin = value."""
    x_origin = 0.5
    y_origin = -1.4
    z_origin = 5.2
    new_origin = (x_origin, y_origin, z_origin)
    mock_dicom_image.origin = new_origin
    assert mock_dicom_image.GetOrigin() == new_origin
    assert mock_dicom_image.origin == mock_dicom_image.GetOrigin()


def test_image_direction_getter(mock_dicom_image: Image):
    """Test Image.direction()."""
    assert mock_dicom_image.direction == mock_dicom_image.GetDirection()


def test_image_direction_setter(mock_dicom_image: Image):
    """Test Image.direction = value."""
    dicom_direction = "0.0\\1.0\\0.0\\-1.0\\0.0\\0.0"
    new_direction = tuple(float(value) for value in dicom_direction.split("\\")) + (0, 0, 1)
    mock_dicom_image.direction = new_direction
    assert mock_dicom_image.GetDirection() == new_direction
    assert mock_dicom_image.direction == mock_dicom_image.GetDirection()


def test_image_size_getter(mock_dicom_image: Image):
    """Test Image.size()."""
    assert mock_dicom_image.size == mock_dicom_image.GetSize()


def test_saved_nifti_file_pixels(mock_dicom_image: Image, tmp_path):
    """Check if the saved nifti file corresponds to the one read by SimpleITK."""
    nifti_file_path = tmp_path / "testfile.nii"
    mock_dicom_image.write(nifti_file_path)
    sitk_image = sitk.ReadImage(nifti_file_path)
    assert np.all(sitk.GetArrayFromImage(sitk_image) == mock_dicom_image.numpy())


@pytest.mark.parametrize("dtype", [None, np.int64])
def test_numpy(dtype, mock_dicom_image: Image):
    """Test numpy array generation from image."""
    if dtype is None:
        image_array = mock_dicom_image.numpy()
    else:
        image_array = mock_dicom_image.numpy(dtype)
    assert isinstance(image_array, np.ndarray)
    # array shape (z_dim, y_dim, x_dim) (reversed from Image.GetSize())
    assert image_array.shape == tuple(reversed(mock_dicom_image.GetSize()))
    assert np.all(image_array == sitk.GetArrayFromImage(mock_dicom_image))


def test_saved_nifti_file_metadata(tmp_path, mock_dicom_image: Image):
    """Check if the saved nifti file metadata corresponds to the one read from the dicom."""
    nifti_file_path = tmp_path / "testfile.nii"
    mock_dicom_image.write(nifti_file_path)
    nifti_image = Image().read(nifti_file_path)
    for key, value in mock_dicom_image.metadata.items():
        assert nifti_image.metadata[key] == value


def test_import_nifti_without_metadata(tmp_path, mock_dicom_image: Image):
    """Test if the import of a nifti file saved without this library succeeds."""
    nifti_file_name = tmp_path / "nifti_image.nii"
    sitk.WriteImage(mock_dicom_image, nifti_file_name)
    nifti_image = Image.read(nifti_file_name)

    assert np.all(nifti_image.numpy() == mock_dicom_image.numpy())

    # check image dimension
    assert int(nifti_image.metadata["dim[0]"]) == 3
    image_dimension = int(nifti_image.metadata["dim[0]"])
    assert np.all(
        [int(nifti_image.metadata[f"dim[{i+1}]"]) for i in range(image_dimension)]
        == list(mock_dicom_image.GetSize())
    )
    # check voxel size
    assert np.allclose(
        [float(nifti_image.metadata[f"pixdim[{i+1}]"]) for i in range(image_dimension)],
        list(mock_dicom_image.GetSpacing()),
        atol=0.001,
    )


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_write_image(file_format, mock_dicom_image: Image, tmp_path):
    """Test if write correctly overrides all write functions."""
    if file_format == "nifti":
        output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    elif file_format == "dicom":
        output_file_name = tmp_path / "dicom" / "dicom_image"
    else:
        raise NotImplementedError

    mock_dicom_image.write(output_file_name)

    if file_format == "nifti":
        new_image = sitk.ReadImage(output_file_name, imageIO="NiftiImageIO")
    elif file_format == "dicom":
        dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(output_file_name))
        new_image = sitk.ReadImage(dicom_images, imageIO="GDCMImageIO")
    else:
        raise ValueError

    assert np.all(sitk.GetArrayFromImage(new_image) == mock_dicom_image.numpy())


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_read_image(file_format, mock_dicom_image: Image, tmp_path):
    """Test if read correctly overrides all read functions."""
    if file_format == "nifti":
        output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    elif file_format == "dicom":
        output_file_name = tmp_path / "dicom" / "dicom_image"
    else:
        raise NotImplementedError

    mock_dicom_image.write(output_file_name)
    new_image = Image.read(output_file_name)

    if file_format == "nifti":
        new_image_reference = sitk.ReadImage(output_file_name, imageIO="NiftiImageIO")
    elif file_format == "dicom":
        new_image_reference, _ = dicom_series.read(output_file_name)
    else:
        raise ValueError

    assert np.all(new_image.numpy() == sitk.GetArrayFromImage(new_image_reference))


def test_default_read_image_metadata(mock_dicom_image: Image, tmp_path):
    """Test default arguments of read."""
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    mock_dicom_image.write(output_file_name)
    new_image_default = Image.read(output_file_name)
    new_image = Image.read(output_file_name, read_metadata=True)

    assert new_image_default.metadata == new_image.metadata


def test_read_image_without_metadata(mock_dicom_image: Image, tmp_path):
    """Test read_metadata argument of read."""
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    mock_dicom_image.write(output_file_name)
    reference_image = sitk.ReadImage(output_file_name)
    new_image = Image.read(output_file_name, read_metadata=False)
    new_image_metadata = {elem: "" for elem in REQUIRED_IMAGE_FIELDS}
    for key in reference_image.GetMetaDataKeys():
        value = reference_image.GetMetaData(key)
        value = format_digit_string(value)
        new_image_metadata[key] = value
    assert new_image.metadata == new_image_metadata


def test_default_write_image_metadata(mock_dicom_image: Image, tmp_path):
    """Test default arguments of write."""
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    default_output_file_name = tmp_path / "nifti_default" / "nifti_image.nii"
    mock_dicom_image.write(default_output_file_name)
    mock_dicom_image.write(output_file_name, write_metadata=True)

    new_image_default = Image().read(default_output_file_name)
    new_image = Image().read(output_file_name)

    assert new_image_default.metadata == new_image.metadata


def test_write_image_without_metadata(mock_dicom_image: Image, tmp_path):
    """Test read_metadata argument of write."""
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    reference_output_file_name = tmp_path / "nifti_default" / "nifti_image.nii"
    reference_output_file_name.parent.mkdir()

    mock_dicom_image.write(output_file_name, write_metadata=False)
    sitk.WriteImage(mock_dicom_image, reference_output_file_name)

    reference_image = Image().read(reference_output_file_name)
    new_image = Image().read(output_file_name)
    assert new_image.metadata == reference_image.metadata


@pytest.mark.parametrize("copy", [True, False])
def test_image_from_array(mock_dicom_image: Image, copy):
    """Test image creation from a numpy array."""
    new_image = Image().from_array(
        mock_dicom_image.numpy(copy=copy),
        spacing=mock_dicom_image.spacing,
        origin=mock_dicom_image.origin,
        direction=mock_dicom_image.direction,
        metadata=mock_dicom_image.metadata,
    )
    assert new_image.size == mock_dicom_image.size
    assert new_image.spacing == mock_dicom_image.spacing
    assert new_image.origin == mock_dicom_image.origin
    assert new_image.direction == mock_dicom_image.direction
    assert np.all(new_image.numpy(copy=copy) == mock_dicom_image.numpy(copy=copy))
    assert new_image.metadata == mock_dicom_image.metadata


@pytest.mark.parametrize("copy", [True, False, None])
def test_image_view_from_array(copy, mock_dicom_image: Image):
    """Test array generation from image."""
    if copy is True:
        mock_dicom_image.numpy(copy=copy)[:] = 0
        np.asarray(mock_dicom_image, copy=copy)[:] = 0
    else:
        with pytest.raises(ValueError):
            mock_dicom_image.numpy(copy=copy)[:] = 0
        with pytest.raises(ValueError):
            np.asarray(mock_dicom_image, copy=copy)[:] = 0


@pytest.mark.parametrize("copy", [True, False, None])
def test_image_view_from_array_same_dtype(copy, mock_dicom_image: Image):
    """Test array generation from image casting the same dtype."""
    image_type = mock_dicom_image.dtype
    if copy is True:
        mock_dicom_image.numpy(dtype=image_type, copy=copy)[:] = 0
        np.asarray(mock_dicom_image, dtype=image_type, copy=copy)[:] = 0
    else:
        with pytest.raises(ValueError):
            mock_dicom_image.numpy(dtype=image_type, copy=copy)[:] = 0
        with pytest.raises(ValueError):
            np.asarray(mock_dicom_image, dtype=image_type, copy=copy)[:] = 0


@pytest.mark.parametrize("copy", [True, False, None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_image_view_from_array_different_dtype(copy, dtype, mock_dicom_image: Image):
    """Test array generation from image casting the same dtype."""
    if copy is True or copy is None:
        mock_dicom_image.numpy(dtype=dtype, copy=copy)[:] = 0
        np.asarray(mock_dicom_image, dtype=dtype, copy=copy)[:] = 0
    else:
        with pytest.raises(ValueError):
            mock_dicom_image.numpy(dtype=dtype, copy=copy)[:] = 0
        with pytest.raises(ValueError):
            np.asarray(mock_dicom_image, dtype=dtype, copy=copy)[:] = 0


@pytest.mark.parametrize("scale", [0.5, 2])
def test_image_resample(scale, mock_dicom_image: Image):
    """Test Image.resample()."""
    new_spacing = np.array(mock_dicom_image.spacing) / scale
    if scale <= 1:
        resampled_image = mock_dicom_image.resample(new_spacing.tolist())
    else:
        # When zooming in, sampling is very dependent
        # on the interpolator and on the fill value.
        resampled_image = mock_dicom_image.resample(
            new_spacing.tolist(), interpolator=sitk.sitkBSpline, default_pixel_value=-400
        )
    assert np.all(
        np.array(resampled_image.GetSize())
        == (np.array(mock_dicom_image.GetSize()) * scale + 1e-14).round().astype(int)
    )
    assert all(resampled_image.spacing == new_spacing)
    # Mean image intensity values should be similar
    assert np.allclose(resampled_image.numpy().mean(), mock_dicom_image.numpy().mean(), rtol=0.009)


def test_image_pad_different_spacing(mock_dicom_image: Image):
    """Test image padding with different voxel spacing."""
    reference_image = Image.read(dicom_ct_path())
    reference_image.spacing = (0.5, 1.2, 4.3)
    assert mock_dicom_image.spacing != reference_image.spacing
    with pytest.raises(ValueError):
        mock_dicom_image.pad(reference_image)


def test_image_pad_different_direction(mock_dicom_image: Image):
    """Test image padding with different direction."""
    reference_image = Image.read(dicom_ct_path())
    reference_image.direction = (1, 0, 0, 0, 0, 1, 0, 1, 0)
    assert mock_dicom_image.direction != reference_image.direction
    with pytest.raises(ValueError):
        mock_dicom_image.pad(reference_image)


@pytest.mark.parametrize("left_shift", [-1, 0, 1])
@pytest.mark.parametrize("right_shift", [-1, 0, 1])
def test_image_pad(left_shift, right_shift):
    """Test Image.pad()."""
    image_spacing = (1, 1, 1)
    image_origin = np.array((0, 0, 0))
    reference_origin = image_origin + left_shift
    image_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    original_shape = (5, 5, 5)
    reference_size = np.array(original_shape) + right_shift
    original_array = np.zeros(original_shape)
    point_coordinate = (2, 2, 2)
    original_array[point_coordinate] = 1
    original_image = Image.from_array(
        original_array,
        spacing=image_spacing,
        origin=tuple(image_origin.tolist()),
        direction=image_direction,
    )
    reference_image = Image().from_array(
        np.zeros(reference_size),
        spacing=image_spacing,
        origin=tuple(reference_origin.tolist()),
        direction=image_direction,
    )
    padded_image = original_image.pad(reference_image)
    new_coordinate = np.array(point_coordinate) - left_shift

    assert padded_image.numpy().shape == reference_image.numpy().shape
    assert padded_image.numpy()[tuple(new_coordinate.tolist())] == 1


@pytest.mark.parametrize(
    "dtype",
    [
        {"image": sitk.sitkInt16, "array": np.int16},
        {"image": sitk.sitkInt32, "array": np.int32},
        {"image": sitk.sitkFloat32, "array": np.float32},
        {"image": sitk.sitkFloat64, "array": np.float64},
        {"image": sitk.sitkComplexFloat64, "array": np.complex128},
    ],
)
def test_image_astype_sitk(dtype, mock_dicom_image: Image):
    """Test image type casting with sitk types."""
    image_array = mock_dicom_image.numpy()
    assert (
        mock_dicom_image.astype(dtype["image"]).numpy().dtype
        == image_array.astype(dtype["array"]).dtype
    )
    assert np.all(mock_dicom_image.astype(dtype["image"]).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [np.int16, np.int32, np.float32, np.float64, np.complex128],
)
def test_image_astype_numpy(dtype, mock_dicom_image: Image):
    """Test image type casting with numpy types."""
    image_array = mock_dicom_image.numpy()
    assert mock_dicom_image.astype(dtype).numpy().dtype == image_array.astype(dtype).dtype
    assert np.all(mock_dicom_image.astype(dtype).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [int, float, complex],
)
def test_image_astype_native(dtype, mock_dicom_image: Image):
    """Test image type casting with native Python types."""
    image_array = mock_dicom_image.numpy()
    assert mock_dicom_image.astype(dtype).numpy().dtype == image_array.astype(dtype).dtype
    assert np.all(mock_dicom_image.astype(dtype).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [np.float16],
)
def test_image_astype_numpy_unsupported(dtype, mock_dicom_image: Image):
    """Test image type casting with unsupported numpy types."""
    with pytest.raises(ValueError):
        mock_dicom_image.astype(dtype)


@pytest.mark.parametrize(
    "coregistration_metric",
    [CoregistrationMetric.correlation, CoregistrationMetric.mutual_information],
)
def test_image_coregistration(coregistration_metric, mock_dicom_image: Image):
    """Coregister images."""
    reference_image = Image.read(dicom_ct_path())
    assert mock_dicom_image.origin == reference_image.origin
    mock_dicom_image.origin = (0, 0, 0)
    np.testing.assert_equal(mock_dicom_image.numpy(), reference_image.numpy())
    coregistered_image = mock_dicom_image.coregister(
        reference_image=reference_image,
        fill_value=-1000,
        coregistration_metric=coregistration_metric,
        seed=1,
        num_threads=1,
    )
    assert mock_dicom_image.origin != reference_image.origin
    np.testing.assert_allclose(coregistered_image.origin, reference_image.origin)
    np.testing.assert_allclose(coregistered_image.direction, reference_image.direction)
    np.testing.assert_allclose(coregistered_image.spacing, reference_image.spacing)
    reference_coregistered_image = Image.read(coregistered_image_path(coregistration_metric.name))
    np.testing.assert_equal(
        coregistered_image.numpy(),
        reference_coregistered_image.numpy(),
    )


@pytest.mark.parametrize(
    "image_type", [sitk.sitkInt16, sitk.sitkFloat32, int, float, np.int16, np.float32]
)
def test_get_image_dtype(image_type, mock_dicom_image: Image):
    """Get datatype from image."""
    input_image = mock_dicom_image.astype(image_type)
    assert sitk_image_dtype(input_image.dtype) == sitk_image_dtype(image_type)


@pytest.fixture
def mock_ct():
    """One-pixel CT used for testing operators."""
    return Image.from_array(
        [[[2]]], origin=(0, 0, 0), spacing=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)
    ).astype(np.int16)


POSSIBLE_DTYPES = [
    int,
    float,
    np.uint16,
    np.uint32,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]
"""Possible datatypes used in operations."""


@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_add_types(value_type, mock_ct: Image):
    """Check output types when adding constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr + other_value
    new_ct = mock_ct + other_value
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_radd_types(value_type, mock_ct: Image):
    """Check output types when adding constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr + other_value
    new_ct = other_value + mock_ct
    assert isinstance(new_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_iadd_types(value_type, mock_ct: Image):
    """Check output types when adding constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr + other_value
    mock_ct += other_value
    assert isinstance(mock_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(mock_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == mock_ct.dtype


@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_sub_types(value_type, mock_ct: Image):
    """Check output types when subtracting constants from images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr - other_value
    new_ct = mock_ct - other_value
    assert isinstance(new_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_isub_types(value_type, mock_ct: Image):
    """Check output types when subtracting constants from images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr - other_value
    mock_ct -= other_value
    assert isinstance(mock_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(mock_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == mock_ct.dtype


@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_mul_types(value_type, mock_ct: Image):
    """Check output types when multiplying constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr * other_value
    new_ct = mock_ct * other_value
    assert isinstance(new_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_rmul_types(value_type, mock_ct: Image):
    """Check output types when multiplying constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr * other_value
    new_ct = other_value * mock_ct
    assert isinstance(new_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_imul_types(value_type, mock_ct: Image):
    """Check output types when multiplying constants to images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr * other_value
    mock_ct *= other_value
    assert isinstance(mock_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(mock_ct))
    if not np.issubdtype(value_type, np.integer):
        assert new_arr.dtype == mock_ct.dtype


@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_truediv_types(value_type, mock_ct: Image):
    """Check output types when dividing constants from images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr / other_value
    new_ct = mock_ct / other_value
    assert isinstance(new_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(new_ct))
    if value_type == np.float32:
        assert new_ct.dtype == np.float64
    else:
        assert new_arr.dtype == new_ct.dtype


@pytest.mark.xfail
@pytest.mark.parametrize("value_type", POSSIBLE_DTYPES)
def test_image_itruediv_types(value_type, mock_ct: Image):
    """Check output types when dividing constants from images."""
    mock_arr = mock_ct.numpy()
    assert mock_ct.dtype == mock_arr.dtype
    other_value = value_type(1)
    new_arr = mock_arr / other_value
    mock_ct /= other_value
    assert isinstance(mock_ct, Image)
    np.testing.assert_equal(new_arr, np.asarray(mock_ct))
    if value_type == np.float32:
        assert mock_ct.dtype == np.float64
    else:
        assert new_arr.dtype == mock_ct.dtype


@pytest.mark.parametrize(
    "image_path",
    [
        Path("IBSI1_CT_phantom") / "CT_00000",
        Path("siemens_mprage_0_dcm"),
    ],
)
def test_image_modalities(image_path, caplog, tmp_path):
    """Test image modality when reading files."""
    image_dir = Path(__file__).parent / "Dicom" / image_path
    image = Image.read(image_dir)
    assert getattr(DicomModality, image.modality.lower()) in SERIES_MODALITIES

    nifti_path = tmp_path / "image.nii.gz"
    image.write(nifti_path, write_metadata=True)
    nifti_image = Image.read(nifti_path)
    assert nifti_image.modality == image.modality

    with caplog.at_level(logging.WARNING):
        nifti_image = Image.read(nifti_path, read_metadata=False)
        assert nifti_image.modality == "CT"
    for record in caplog.records:
        assert record.levelname == "WARNING"
        assert record.message == "Image modality must be defined."


def test_empty_image_required_metadata_fields():
    """Test if an empty image has all required metadata."""
    image = Image()
    assert len(image.metadata) == len(REQUIRED_IMAGE_FIELDS)
    for key in image.metadata:
        assert key in REQUIRED_IMAGE_FIELDS


def test_image_patient_id(mock_ct: Image):
    """Test if the PatientID has the correct type."""
    assert isinstance(mock_ct.patient_id, str)


def test_image_ids(mock_dicom_image: Image):
    """Test if ids are stored correctly."""
    ds = dcmread(list(dicom_ct_path().glob("*.dcm"))[0])
    assert ds.PatientID == mock_dicom_image.patient_id
    assert isinstance(mock_dicom_image.patient_id, str)
    assert ds["StudyInstanceUID"].value == mock_dicom_image.study_instance_uid
    assert isinstance(mock_dicom_image.study_instance_uid, str)
