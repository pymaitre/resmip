"""Test module for image.py"""

import numpy as np
import pytest
import SimpleITK as sitk

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


def test_saved_nifti_file_pixels(tmp_path):
    """Check if the saved nifti file corresponds to the one read by SimpleITK."""
    image = Image().read_image(dicom_ct_path())
    nifti_file_path = tmp_path / "testfile.nii"
    image.write_image(nifti_file_path)
    sitk_image = sitk.ReadImage(nifti_file_path)
    assert np.all(sitk.GetArrayFromImage(sitk_image) == sitk.GetArrayFromImage(image))


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

    assert np.all(sitk.GetArrayFromImage(nifti_image) == sitk.GetArrayFromImage(input_image))

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

    assert np.all(sitk.GetArrayFromImage(new_image) == sitk.GetArrayFromImage(input_image))


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

    assert np.all(sitk.GetArrayFromImage(new_image) == sitk.GetArrayFromImage(new_image_reference))


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
