"""Test module for image.py"""

import numpy as np
import pytest
import SimpleITK as sitk

from srmip import DICOM_FIELDS
from srmip.dicom_nifti_conversion.series import read_dicom_series
from srmip.image.image import Image
from srmip.utils import format_digit_string

from .utils import dicom_ct_path


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


def test_saved_nifti_file_pixels(tmp_path):
    """Check if the saved nifti file corresponds to the one read by SimpleITK."""
    image = Image().read_image(dicom_ct_path())
    nifti_file_path = tmp_path / "testfile.nii"
    image.write_image(nifti_file_path)
    sitk_image = sitk.ReadImage(nifti_file_path)
    assert np.all(sitk.GetArrayFromImage(sitk_image) == image.numpy())


def test_numpy():
    """Test numpy array generation from image."""
    dicom_image = Image.read_image(dicom_ct_path())
    assert isinstance(dicom_image.numpy(), np.ndarray)
    # array shape (z_dim, y_dim, x_dim) (reversed from Image.GetSize())
    assert dicom_image.numpy().shape == tuple(reversed(dicom_image.GetSize()))
    assert np.all(dicom_image.numpy() == sitk.GetArrayFromImage(dicom_image))


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
        new_image_reference, _ = read_dicom_series(output_file_name)

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
