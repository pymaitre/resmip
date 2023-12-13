"""Read and write dicom files."""

from pathlib import Path
from typing import Union

import numpy as np
import SimpleITK as sitk


def read_dicom_series(dicom_series_directory_path: Union[str, Path]) -> None:
    dicom_series_files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
        str(dicom_series_directory_path)
    )
    slices_number = len(dicom_series_files)
    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_series_reader.SetFileNames(dicom_series_files)
    dicom_series_reader.MetaDataDictionaryArrayUpdateOn()
    dicom_series_reader.LoadPrivateTagsOn()
    dicom_image = dicom_series_reader.Execute()
    slice_0_metadata = {}
    slice_1_metadata = {}
    series_metadata = {}
    for k in dicom_series_reader.GetMetaDataKeys(0):
        v = dicom_series_reader.GetMetaData(0, k)
        slice_0_metadata[k] = v
        series_metadata[k] = v
    for k in dicom_series_reader.GetMetaDataKeys(1):
        v = dicom_series_reader.GetMetaData(1, k)
        slice_1_metadata[k] = v
    for (key, value1), (_, value2) in zip(slice_0_metadata.items(), slice_1_metadata.items()):
        if value1 != value2:
            print(key, value1, value2)
    print(sitk.GetArrayFromImage(dicom_image).shape)
    for slice_dependent_field in SLICE_DEPENDENT_FIELDS.keys():
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


SLICE_DEPENDENT_FIELDS = {
    "0008|0018": "SOPInstanceUID",
    "0020|0013": "InstanceNumber",
    "0020|0032": "ImagePositionPatient",
}

# TODO: This part must be removed!
# study_path = Path(__file__).parents[1] / "tests" / "Dicom" / "IBSI1_CT_phantom" / "image"
# print(study_path)
# read_dicom_series(study_path)
