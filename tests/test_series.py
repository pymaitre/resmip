"""Test module for series.py"""

import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

import srmip.dicom_utils.series as dicom_series
from srmip import read_image, write_image
from srmip.dicom_utils.constants import DICOM_FIELDS, SERIES_DEPENDENT_FIELDS
from srmip.image.image import Image

from .utils import dicom_ct_path


def compare_dicom_pixels(image: Image, dicom_path: Path):
    """Compare dicom pixel values between slices."""
    image_array = image.numpy()
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
    image, image_metadata = dicom_series.read(dicom_ct_path())
    image_array = sitk.GetArrayFromImage(image) + 1000
    new_image = Image(sitk.GetImageFromArray(image_array))
    new_image.metadata = image_metadata
    compare_dicom_pixels(new_image, dicom_ct_path())


def test_saved_dicom_series_pixels(tmp_path):
    """Check if the saved Dicom series pixel grid is saved correctly."""
    input_image = Image().read_image(dicom_ct_path())
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


def test_read_image_from_main():
    """Check if the read_image function behaves as expected."""
    input_image_class = Image().read_image(dicom_ct_path())
    input_image_main = read_image(dicom_ct_path())
    assert input_image_main == input_image_class


def test_write_image_from_main(tmp_path):
    """Check if the read_image function behaves as expected."""
    input_image = Image().read_image(dicom_ct_path())

    input_image.write_image(tmp_path / "from_class")
    write_image(input_image, tmp_path / "from_main")

    image_from_class = Image().read_image(tmp_path / "from_class")
    image_from_main = Image().read_image(tmp_path / "from_main")

    assert image_from_class == image_from_main


def test_multiple_modalities_in_same_folder():
    """Read dicom CT image with other modalities in the same directory."""
    multiple_modalities_path = Path(__file__).parent / "Dicom" / "dicompyler_img"
    input_image_files = dicom_series.get_series_dicom_files(multiple_modalities_path)
    assert input_image_files == (str(multiple_modalities_path / "ct.0.dcm"),)


def test_read_series_in_empty_folder(tmp_path):
    """Read dicom CT image in an empty directory."""
    input_image_files = dicom_series.get_series_dicom_files(tmp_path)
    assert input_image_files == tuple()


def test_spacing_single_slice_series(caplog):
    """Test voxel spacing for series with only one z slice."""
    image_path = Path(__file__).parent / "Dicom" / "dicompyler_img"
    image = Image().read_image(image_path)
    sitk_image = sitk.ReadImage(str(image_path / "ct.0.dcm"))
    assert image.metadata[DICOM_FIELDS["Modality"]] == "CT"
    assert sitk_image.GetSpacing() == image.spacing
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Only 1 slice detected. Setting z voxel spacing to 1 mm." in caplog.text
