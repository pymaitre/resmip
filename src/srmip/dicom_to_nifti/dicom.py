"""Read and write dicom files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import SimpleITK as sitk

FileNameType = Union[str, Path]
"""Types used int the classes and functions for file names."""


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

    def metadata_file_name(self, filename: FileNameType) -> Path:
        """
        Generate the filename for the metadata.

        Defaults a json file with same name of the output image file (filename).
        The json filename is prepended with a "." to make it hidden.

        :param filename: name of the output image file name.
        :return: Path of the json metadata file.
        """
        filename = Path(filename)
        return filename.parent / f".{filename.stem}.json"

    def write_nifti(self, filename: FileNameType) -> None:
        """Save nifti file (and metadata)."""
        filename = Path(filename)
        sitk.WriteImage(self, str(filename))
        serialized_metadata = json.dumps(self.metadata)
        self.metadata_file_name(filename).write_text(serialized_metadata)

    @staticmethod
    def read_nifti(filename: FileNameType) -> Image:
        filename = Path(filename)
        new_image = Image(sitk.ReadImage(str(filename)))
        serialized_metadata = new_image.metadata_file_name(filename).read_text()
        new_image.metadata = serialized_metadata
        return new_image


def read_dicom_series(dicom_series_directory_path: FileNameType) -> None:
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
    for k in dicom_series_reader.GetMetaDataKeys(0):
        v = dicom_series_reader.GetMetaData(0, k)
        series_metadata[k] = v
    for slice_dependent_field in SLICE_DEPENDENT_FIELDS:
        try:
            series_metadata.pop(slice_dependent_field)
        except KeyError:
            # if the key is not present in the image, we must find a way to keep track of this information
            pass

    instance_numbers = np.zeros(slices_number, dtype=int)
    image_position_patients = np.zeros((slices_number, 3), dtype=float)
    for i in range(slices_number):
        instance_number = dicom_series_reader.GetMetaData(i, "0020|0013")
        image_position_patient = dicom_series_reader.GetMetaData(i, "0020|0032")
        instance_numbers[i] = instance_number
        image_position_patients[i] = image_position_patient.split("\\")
    slices_indexes = {
        "min": instance_numbers.min(),
        "max": instance_numbers.max(),
    }  # TODO: we must store this information in the image
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
            series_metadata[key] = value  # TODO: we should be able to go back to np.array somehow
        dicom_series.SetMetaData(key, value)
    dicom_series.metadata = series_metadata
    return dicom_series


SLICE_DEPENDENT_FIELDS = {
    "0008|0018": "SOPInstanceUID",
    "0020|0013": "InstanceNumber",
    "0020|0032": "ImagePositionPatient",
}

# TODO: This part must be removed!
# study_path = Path(__file__).parents[1] / "tests" / "Dicom" / "IBSI1_CT_phantom" / "image"
# print(study_path)
# read_dicom_series(study_path)
