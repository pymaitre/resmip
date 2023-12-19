"""Test module for dicom.py"""

import json
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from srmip.dicom_to_nifti.constants import DICOM_FIELDS, SERIES_DEPENDENT_FIELDS
from srmip.dicom_to_nifti.dicom import Image, read_dicom_series


def dicom_ct_path() -> Path:
    """Path of the test Dicom CT."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "image"


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


def compare_dicom_pixels(image: Image, dicom_path: Path):
    """Compare dicom pixel values between slices."""
    image_array = sitk.GetArrayFromImage(image)
    min_instance_number = np.array(json.loads(image.metadata["slice_indexes"])).max()
    for dicom_file in dicom_path.glob("*.dcm"):
        dataset = pydicom.dcmread(dicom_file)
        if dataset["Modality"].value != "CT":
            continue
        slice_index = min_instance_number - dataset["InstanceNumber"].value
        pixel_array = np.frombuffer(dataset.PixelData, dtype=np.int16).reshape(
            (dataset["Rows"].value, dataset["Columns"].value)
        )
        assert np.all(pixel_array == image_array[slice_index, :, :])


def test_dicom_image_pixel_array():
    """Check if the pixel grid read by SimpleITK corresponds to the one in the Dicom files."""
    image, image_metadata = read_dicom_series(dicom_ct_path())
    image_array = sitk.GetArrayFromImage(image) + 1000
    new_image = Image(sitk.GetImageFromArray(image_array))
    new_image.metadata = image_metadata
    compare_dicom_pixels(new_image, dicom_ct_path())


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


def test_saved_dicom_series_pixels(tmp_path):
    """Check if the saved Dicom series pixel grid is saved correctly."""
    input_image = Image.read_image(dicom_ct_path())
    input_image.write_image(tmp_path)

    compare_dicom_pixels(input_image, tmp_path)


def test_saved_dicom_series_patient_data(tmp_path):
    """Check if dicom header values are the same."""
    input_image = Image.read_image(dicom_ct_path())
    input_image.write_image(tmp_path)

    for dicom_file in tmp_path.glob("*.dcm"):
        dataset = pydicom.dcmread(dicom_file)
        for name, tag in DICOM_FIELDS.items():
            if name in SERIES_DEPENDENT_FIELDS:
                continue
            if tag in input_image.metadata:
                if dataset[name].VR == "DS":  # DecimalString
                    # Empty field in the header
                    if input_image.metadata[tag] == "":
                        assert dataset[name].value is None
                        continue
                    try:
                        assert float(dataset[name].value) == float(input_image.metadata[tag])
                    except TypeError:  # list[float]
                        elements = input_image.metadata[tag].split("\\")
                        for i, element in enumerate(dataset[name].value):
                            assert float(element) == float(elements[i])
                else:
                    assert dataset[name].value == input_image.metadata[tag]


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
