"""Test module for image.py"""

import numpy as np
import pytest
import SimpleITK as sitk

import resmip.dicom_utils.series as dicom_series
from resmip import DICOM_FIELDS
from resmip.image.image import Image
from resmip.utils import format_digit_string

from .utils import coregistered_image_path, dicom_ct_path


def test_metadata_is_unique():
    """Test if setting one Image's metadata does not touch another series."""
    new_image1 = Image()
    new_image2 = Image()

    assert new_image1.metadata == {}
    assert new_image2.metadata == {}

    new_image1.metadata["a"] = 0

    assert new_image1.metadata == {"a": 0}
    assert new_image2.metadata == {}


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_metadata_contains_only_strings(file_format, tmp_path):
    """Check that all elements in the read Dicom header are python strings."""
    dicom_image = Image.read_image(dicom_ct_path())
    if file_format == "nifti":
        nifti_image_path = tmp_path / "image.nii"
        dicom_image.write_image(nifti_image_path)
        dicom_image = Image().read_image(nifti_image_path)

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
    metadata_file_name = Image().metadata_file_name(given_nifti_file_name)
    assert metadata_file_name == tmp_path / f".{nifti_file_name.stem}.json"


def test_image_spacing_getter():
    """Test Image.spacing()."""
    dicom_image = Image.read_image(dicom_ct_path())
    assert dicom_image.spacing == dicom_image.GetSpacing()


def test_image_spacing_setter():
    """Test Image.spacing = value."""
    dicom_image = Image.read_image(dicom_ct_path())
    x_spacing = 0.5
    y_spacing = 1
    z_spacing = 5.2
    xy_spacing = f"{x_spacing}\\{y_spacing}"
    new_spacing = (x_spacing, y_spacing, z_spacing)
    dicom_image.spacing = new_spacing
    assert dicom_image.GetSpacing() == new_spacing
    assert dicom_image.spacing == dicom_image.GetSpacing()
    assert dicom_image.metadata[DICOM_FIELDS["PixelSpacing"]] == xy_spacing
    assert dicom_image.metadata[DICOM_FIELDS["SliceThickness"]] == str(z_spacing)


def test_image_origin_getter():
    """Test Image.origin()."""
    dicom_image = Image.read_image(dicom_ct_path())
    assert dicom_image.origin == dicom_image.GetOrigin()


def test_image_origin_setter():
    """Test Image.origin = value."""
    dicom_image = Image.read_image(dicom_ct_path())
    x_origin = 0.5
    y_origin = -1.4
    z_origin = 5.2
    new_origin = (x_origin, y_origin, z_origin)
    dicom_image.origin = new_origin
    assert dicom_image.GetOrigin() == new_origin
    assert dicom_image.origin == dicom_image.GetOrigin()


def test_image_direction_getter():
    """Test Image.direction()."""
    dicom_image = Image.read_image(dicom_ct_path())
    assert dicom_image.direction == dicom_image.GetDirection()


def test_image_direction_setter():
    """Test Image.direction = value."""
    dicom_image = Image.read_image(dicom_ct_path())
    dicom_direction = "0.0\\1.0\\0.0\\-1.0\\0.0\\0.0"
    new_direction = tuple(float(value) for value in dicom_direction.split("\\")) + (0, 0, 1)
    dicom_image.direction = new_direction
    assert dicom_image.GetDirection() == new_direction
    assert dicom_image.direction == dicom_image.GetDirection()
    assert dicom_image.metadata[DICOM_FIELDS["ImageOrientationPatient"]] == dicom_direction


def test_image_size_getter():
    """Test Image.size()."""
    dicom_image = Image.read_image(dicom_ct_path())
    assert dicom_image.size == dicom_image.GetSize()


def test_saved_nifti_file_pixels(tmp_path):
    """Check if the saved nifti file corresponds to the one read by SimpleITK."""
    image = Image().read_image(dicom_ct_path())
    nifti_file_path = tmp_path / "testfile.nii"
    image.write_image(nifti_file_path)
    sitk_image = sitk.ReadImage(nifti_file_path)
    assert np.all(sitk.GetArrayFromImage(sitk_image) == image.numpy())


@pytest.mark.parametrize("dtype", [None, np.int64])
def test_numpy(dtype):
    """Test numpy array generation from image."""
    dicom_image = Image.read_image(dicom_ct_path())
    if dtype is None:
        image_array = dicom_image.numpy()
    else:
        image_array = dicom_image.numpy(dtype)
    assert isinstance(image_array, np.ndarray)
    # array shape (z_dim, y_dim, x_dim) (reversed from Image.GetSize())
    assert image_array.shape == tuple(reversed(dicom_image.GetSize()))
    assert np.all(image_array == sitk.GetArrayFromImage(dicom_image))


def test_saved_nifti_file_metadata(tmp_path):
    """
    Check if the saved nifti file metadata corresponds to the one read from the dicom.
    """
    dicom_image = Image.read_image(dicom_ct_path())
    nifti_file_path = tmp_path / "testfile.nii"
    dicom_image.write_image(nifti_file_path)
    nifti_image = Image().read_image(nifti_file_path)
    for key, value in dicom_image.metadata.items():
        assert nifti_image.metadata[key] == value


def test_import_nifti_without_metadata(tmp_path):
    """Test if the import of a nifti file saved without this library succeeds."""
    input_image = Image.read_image(dicom_ct_path())
    nifti_file_name = tmp_path / "nifti_image.nii"
    sitk.WriteImage(input_image, nifti_file_name)
    nifti_image = Image().read_image(nifti_file_name)

    assert np.all(nifti_image.numpy() == input_image.numpy())

    # check image dimension
    assert int(nifti_image.metadata["dim[0]"]) == 3
    image_dimension = int(nifti_image.metadata["dim[0]"])
    assert np.all(
        [int(nifti_image.metadata[f"dim[{i+1}]"]) for i in range(image_dimension)]
        == list(input_image.GetSize())
    )
    # check voxel size
    assert np.allclose(
        [float(nifti_image.metadata[f"pixdim[{i+1}]"]) for i in range(image_dimension)],
        list(input_image.GetSpacing()),
        atol=0.001,
    )


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_write_image(file_format, tmp_path):
    """Test if write_image correctly overrides all write functions."""
    input_image = Image().read_image(dicom_ct_path())
    if file_format == "nifti":
        output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    elif file_format == "dicom":
        output_file_name = tmp_path / "dicom" / "dicom_image"
    else:
        raise NotImplementedError

    input_image.write_image(output_file_name)

    if file_format == "nifti":
        new_image = sitk.ReadImage(output_file_name, imageIO="NiftiImageIO")
    elif file_format == "dicom":
        dicom_images = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(output_file_name))
        new_image = sitk.ReadImage(dicom_images, imageIO="GDCMImageIO")
    else:
        raise ValueError

    assert np.all(sitk.GetArrayFromImage(new_image) == input_image.numpy())


@pytest.mark.parametrize("file_format", ["dicom", "nifti"])
def test_read_image(file_format, tmp_path):
    """Test if read_image correctly overrides all read functions."""
    input_image = Image().read_image(dicom_ct_path())
    if file_format == "nifti":
        output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    elif file_format == "dicom":
        output_file_name = tmp_path / "dicom" / "dicom_image"
    else:
        raise NotImplementedError

    input_image.write_image(output_file_name)
    new_image = Image().read_image(output_file_name)

    if file_format == "nifti":
        new_image_reference = sitk.ReadImage(output_file_name, imageIO="NiftiImageIO")
    elif file_format == "dicom":
        new_image_reference, _ = dicom_series.read(output_file_name)
    else:
        raise ValueError

    assert np.all(new_image.numpy() == sitk.GetArrayFromImage(new_image_reference))


def test_default_read_image_metadata(tmp_path):
    """Test default arguments of read_image."""
    input_image = Image().read_image(dicom_ct_path())
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    input_image.write_image(output_file_name)
    new_image_default = Image().read_image(output_file_name)
    new_image = Image().read_image(output_file_name, read_metadata=True)

    assert new_image_default.metadata == new_image.metadata


def test_read_image_without_metadata(tmp_path):
    """Test read_metadata argument of read_image."""
    input_image = Image().read_image(dicom_ct_path())
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    input_image.write_image(output_file_name)
    reference_image = sitk.ReadImage(output_file_name)
    new_image = Image().read_image(output_file_name, read_metadata=False)
    new_image_metadata = {}
    for key in reference_image.GetMetaDataKeys():
        value = reference_image.GetMetaData(key)
        value = format_digit_string(value)
        new_image_metadata[key] = value
    assert new_image.metadata == new_image_metadata


def test_default_write_image_metadata(tmp_path):
    """Test default arguments of write_image."""
    input_image = Image().read_image(dicom_ct_path())
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    default_output_file_name = tmp_path / "nifti_default" / "nifti_image.nii"
    input_image.write_image(default_output_file_name)
    input_image.write_image(output_file_name, write_metadata=True)

    new_image_default = Image().read_image(default_output_file_name)
    new_image = Image().read_image(output_file_name)

    assert new_image_default.metadata == new_image.metadata


def test_write_image_without_metadata(tmp_path):
    """Test read_metadata argument of write_image."""
    input_image = Image().read_image(dicom_ct_path())
    output_file_name = tmp_path / "nifti" / "nifti_image.nii"
    reference_output_file_name = tmp_path / "nifti_default" / "nifti_image.nii"
    reference_output_file_name.parent.mkdir()

    input_image.write_image(output_file_name, write_metadata=False)
    sitk.WriteImage(input_image, reference_output_file_name)

    reference_image = Image().read_image(reference_output_file_name)
    new_image = Image().read_image(output_file_name)
    assert new_image.metadata == reference_image.metadata


@pytest.mark.parametrize("view", [True, False])
def test_image_from_array(view):
    """Test image creation from a numpy array."""
    input_image = Image().read_image(dicom_ct_path())
    new_image = Image().from_array(
        input_image.numpy(view=view),
        spacing=input_image.spacing,
        origin=input_image.origin,
        direction=input_image.direction,
        metadata=input_image.metadata,
    )
    assert new_image.GetSize() == input_image.GetSize()
    assert new_image.spacing == input_image.spacing
    assert new_image.origin == input_image.origin
    assert new_image.direction == input_image.direction
    assert np.all(new_image.numpy(view=view) == input_image.numpy(view=view))
    assert new_image.metadata == input_image.metadata


@pytest.mark.parametrize("view", [True, False])
def test_image_view_from_array(view):
    """Test array generation from image."""
    input_image = Image.read_image(dicom_ct_path())
    if view:
        with pytest.raises(ValueError):
            input_image.numpy(view=view)[:] = 0
    else:
        input_image.numpy(view=view)[:] = 0


@pytest.mark.parametrize("scale", [0.5, 2])
def test_image_resample(scale):
    """Test Image.resample()."""
    input_image = Image().read_image(dicom_ct_path())
    new_spacing = np.array(input_image.spacing) / scale
    if scale <= 1:
        resampled_image = input_image.resample(new_spacing.tolist())
    else:
        # When zooming in, sampling is very dependent
        # on the interpolator and on the fill value.
        resampled_image = input_image.resample(
            new_spacing.tolist(), interpolator=sitk.sitkBSpline, default_pixel_value=-400
        )
    assert np.all(
        np.array(resampled_image.GetSize())
        == (np.array(input_image.GetSize()) * scale + 1e-14).round().astype(int)
    )
    assert all(resampled_image.spacing == new_spacing)
    # Mean image intensity values should be similar
    assert np.allclose(resampled_image.numpy().mean(), input_image.numpy().mean(), rtol=0.009)


def test_image_pad_different_spacing():
    """Test image padding with different voxel spacing."""
    input_image = Image().read_image(dicom_ct_path())
    reference_image = Image()
    reference_image.spacing = (0.5, 1.2, 4.3)
    assert input_image.spacing != reference_image.spacing
    with pytest.raises(ValueError):
        input_image.pad(reference_image)


def test_image_pad_different_direction():
    """Test image padding with different direction."""
    input_image = Image().read_image(dicom_ct_path())
    print(input_image.direction)
    reference_image = Image().read_image(dicom_ct_path())
    reference_image.direction = (1, 0, 0, 0, 0, 1, 0, 1, 0)
    assert input_image.direction != reference_image.direction
    with pytest.raises(ValueError):
        input_image.pad(reference_image)


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
    original_image = Image().from_array(
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
def test_image_astype_sitk(dtype):
    """Test image type casting with sitk types."""
    input_image = Image().read_image(dicom_ct_path())
    image_array = input_image.numpy()

    assert (
        input_image.astype(dtype["image"]).numpy().dtype == image_array.astype(dtype["array"]).dtype
    )
    assert np.all(input_image.astype(dtype["image"]).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [np.int16, np.int32, np.float32, np.float64, np.complex128],
)
def test_image_astype_numpy(dtype):
    """Test image type casting with numpy types."""
    input_image = Image().read_image(dicom_ct_path())
    image_array = input_image.numpy()

    assert input_image.astype(dtype).numpy().dtype == image_array.astype(dtype).dtype
    assert np.all(input_image.astype(dtype).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [int, float, complex],
)
def test_image_astype_native(dtype):
    """Test image type casting with native Python types."""
    input_image = Image().read_image(dicom_ct_path())
    image_array = input_image.numpy()

    assert input_image.astype(dtype).numpy().dtype == image_array.astype(dtype).dtype
    assert np.all(input_image.astype(dtype).numpy() == image_array)


@pytest.mark.parametrize(
    "dtype",
    [np.float16],
)
def test_image_astype_numpy_unsupported(dtype):
    """Test image type casting with unsupported numpy types."""
    input_image = Image().read_image(dicom_ct_path())

    with pytest.raises(ValueError):
        input_image.astype(dtype)


def test_image_coregistration():
    """Coregister images."""
    input_image = Image().read_image(dicom_ct_path())
    reference_image = Image().read_image(dicom_ct_path())
    assert input_image.origin == reference_image.origin
    input_image.origin = (0, 0, 0)
    np.testing.assert_equal(input_image.numpy(), reference_image.numpy())
    coregistered_image = input_image.coregister(
        reference_image=reference_image, fill_value=-1000, seed=1, num_threads=1
    )
    assert input_image.origin != reference_image.origin
    np.testing.assert_allclose(coregistered_image.origin, reference_image.origin)
    np.testing.assert_allclose(coregistered_image.direction, reference_image.direction)
    np.testing.assert_allclose(coregistered_image.spacing, reference_image.spacing)
    reference_coregistered_image = Image().read_image(coregistered_image_path())
    np.testing.assert_equal(
        coregistered_image.numpy(),
        reference_coregistered_image.numpy(),
    )


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_sum(factor):
    """Add constant factor to image pixels."""
    input_image = Image().read_image(dicom_ct_path())
    summed_image = input_image + factor
    assert isinstance(summed_image, Image)
    assert np.all(summed_image.numpy() == input_image.numpy() + factor)
    assert len(summed_image.metadata) == len(input_image.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_subtract(factor):
    """Remove constant factor to image pixels."""
    input_image = Image().read_image(dicom_ct_path())
    summed_image = input_image - factor
    assert isinstance(summed_image, Image)
    assert np.all(summed_image.numpy() == input_image.numpy() - factor)
    assert len(summed_image.metadata) == len(input_image.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_multiply(factor):
    """Multiply constant factor to image pixels."""
    input_image = Image().read_image(dicom_ct_path())
    summed_image = input_image * factor
    assert isinstance(summed_image, Image)
    assert np.all(summed_image.numpy() == input_image.numpy() * factor)
    assert len(summed_image.metadata) == len(input_image.metadata)


@pytest.mark.parametrize("factor", [-1, 0.2, 5])
def test_image_divide(factor):
    """Divide constant factor to image pixels."""
    input_image = Image().read_image(dicom_ct_path())
    summed_image = input_image / factor
    assert isinstance(summed_image, Image)
    assert np.all(summed_image.numpy() == input_image.numpy() / factor)
    assert len(summed_image.metadata) == len(input_image.metadata)
