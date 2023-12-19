"""Read and write dicom files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pydicom
import SimpleITK as sitk

from srmip.dicom_to_nifti.constants import (
    DICOM_FIELDS,
    SERIES_DEPENDENT_FIELDS,
    SLICE_DEPENDENT_FIELDS,
)
from srmip.utils import PathLike, format_digit_string


class Image(sitk.Image):
    """Wrapper class of SimpleITK.Image with support to headers."""

    _metadata: Dict[str, str]
    """Dictionary containing metadata."""

    def __init__(self, *args):
        """Call sitk.Image constructor a create an empty dictionary for the header."""
        super().__init__(*args)
        self._metadata = {}

    @property
    def metadata(self) -> Dict[str, str]:
        """Dicom header combined with other metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    def metadata_file_name(self, filename: PathLike) -> Path:
        """
        Generate the filename for the metadata.

        Defaults a json file with same name of the output image file (filename).
        The json filename is prepended with a "." to make it hidden.

        :param filename: name of the output image file name.
        :type filename: PathLike
        :return: Path of the json metadata file.
        :rtype: Path
        """
        filename = Path(filename)
        return filename.parent / f".{filename.stem}.json"

    @staticmethod
    def read_image(filename: PathLike) -> Image:
        """
        Load image file (and metadata).

        The image format is automatically determined from filename's suffix.
        :param filename: Name of the file. If filename is a directory,
            the writer assumes to write a Dicom series. Otherwise, it assumes a metatadata
            file with the following format exists: f".{filename.stem}.json".
        :type filename: PathLike
        :return: Image and metadata.
        :rtype: Image
        """
        filename = Path(filename)
        if filename.is_dir():
            return read_dicom_series(filename)
        new_image = Image(sitk.ReadImage(filename))
        if new_image.metadata_file_name(filename).exists():
            serialized_metadata = new_image.metadata_file_name(filename).read_text()
            series_metadata = json.loads(serialized_metadata)
        else:
            series_metadata = {}
        for key in new_image.GetMetaDataKeys():
            value = new_image.GetMetaData(key)
            value = format_digit_string(value)
            series_metadata[key] = value
        new_image.metadata = series_metadata
        return new_image

    def write_image(self, filename: PathLike) -> None:
        """
        Save image file (and metadata).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.
        :param filename: Name of the file. If filename is a directory,
            the writer assumes to write a Dicom series.
        :type filename: PathLike
        """
        filename = Path(filename)
        if not filename.exists() and filename.suffix == "":
            filename.mkdir(parents=True, exist_ok=True)
        if filename.is_dir():
            write_dicom_series(self, filename)
            return
        filename.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self, filename)
        serialized_metadata = json.dumps(self.metadata)
        self.metadata_file_name(filename).write_text(serialized_metadata)


def read_dicom_series(dicom_series_directory_path: PathLike) -> Image:
    """
    Read Dicom series from file.

    :param dicom_series_directory_path: Path of the directory containing the Dicom Series.
    :type dicom_series_directory_path: PathLike
    :return: Image and metadata.
    :type: Image
    """
    dicom_series_files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
        str(dicom_series_directory_path)
    )
    slices_number = len(dicom_series_files)
    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_series_reader.SetFileNames(dicom_series_files)
    dicom_series_reader.MetaDataDictionaryArrayUpdateOn()
    dicom_series_reader.LoadPrivateTagsOn()
    dicom_series = Image(dicom_series_reader.Execute())
    series_metadata = {}
    for key in dicom_series_reader.GetMetaDataKeys(0):
        value = dicom_series_reader.GetMetaData(0, key)
        value = format_digit_string(value)
        series_metadata[key] = value
    for slice_dependent_field in SLICE_DEPENDENT_FIELDS:
        try:
            series_metadata.pop(DICOM_FIELDS[slice_dependent_field])
        except KeyError:
            # if the key is not present in the image, we must find a way
            # to keep track of this information
            pass

    instance_numbers = np.zeros(slices_number, dtype=int)
    image_position_patients = np.zeros((slices_number, 3), dtype=float)
    for i in range(slices_number):
        instance_number = dicom_series_reader.GetMetaData(i, DICOM_FIELDS["InstanceNumber"])
        image_position_patient = dicom_series_reader.GetMetaData(
            i, DICOM_FIELDS["ImagePositionPatient"]
        )
        instance_numbers[i] = instance_number
        image_position_patients[i] = image_position_patient.split("\\")
    slices_indexes = {
        "min": instance_numbers.min(),
        "max": instance_numbers.max(),
    }
    assert (
        slices_indexes["max"] - slices_indexes["min"] + 1 == slices_number
    )  # otherwise, throw an error (missing / duplicated slices)
    # saved as numpy array for convenience, but np.arrays cannot be serialized!
    series_metadata["slice_indexes"] = instance_numbers
    series_metadata["image_positions"] = image_position_patients
    for key, value in series_metadata.items():
        # numpy arrays must be serialized to strings
        if not isinstance(value, str):
            value = json.dumps(value.tolist())
            series_metadata[key] = value
        dicom_series.SetMetaData(key, value)
    dicom_series.metadata = series_metadata
    return dicom_series


def write_dicom_series(image: Image, save_path: PathLike) -> None:
    """
    Save the image as a Dicom series.

    :param image: image object to be saved.
    :type image: Image
    :param save_path: Path where the image is saved. save_path must be a directory.
        If it does not already exist, a new dicrectory is created.
    :type save_path: PathLike
    """
    save_path = Path(save_path)
    # the following raises an error if save_path is an existing non-directory file
    save_path.mkdir(parents=True, exist_ok=True)
    series_writer = sitk.ImageFileWriter()
    series_writer.KeepOriginalImageUIDOn()

    instance_numbers = np.array(json.loads(image.metadata["slice_indexes"]))

    image_metadata = {}
    for dicom_tag in DICOM_FIELDS.values():
        if dicom_tag in image.metadata:
            image_metadata[dicom_tag] = image.metadata[dicom_tag]

    for series_dependent_field in SERIES_DEPENDENT_FIELDS:
        image_metadata[DICOM_FIELDS[series_dependent_field]] = pydicom.uid.generate_uid()

    # Do we want to round it back to the value of the Dicom or do we want
    # to keep the value computed by SimpleITK? The pixel grid on the Dicom file
    # is identical to the generated one.
    # Maybe it's safer to round it.
    # image_metadata["0018|0050"] = str(image.GetSpacing()[2])
    rounded_z_spacing = float(f"{image.GetSpacing()[2]:.3e}")
    image_metadata[DICOM_FIELDS["SliceThickness"]] = str(rounded_z_spacing)
    image_metadata[DICOM_FIELDS["SpacingBetweenSlices"]] = str(rounded_z_spacing)

    for i in range(image.GetDepth()):
        image_slice = image[:, :, i]

        slice_metadata = image_metadata.copy()
        slice_metadata[DICOM_FIELDS["InstanceNumber"]] = instance_numbers[i]

        for key, value in slice_metadata.items():
            image_slice.SetMetaData(key, str(value))

        # maybe these are not needed
        # image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        # image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

        image_slice.SetMetaData(
            DICOM_FIELDS["ImagePositionPatient"],
            "\\".join(map(str, image.TransformIndexToPhysicalPoint((0, 0, i)))),
        )  # Image Position (Patient)
        # these are set using the dicom header, but maybe we need them
        # in case we import directly nifti files
        # image_slice.SetMetaData("0020|0013", str(i))  # Instance Number
        # image_slice.SetMetaData(
        #     "0020|1041", str(image.TransformIndexToPhysicalPoint((0, 0, i))[2])
        # )  # Slice Location

        series_writer.SetFileName(str(save_path / f"{i}.dcm"))
        series_writer.Execute(image_slice)
