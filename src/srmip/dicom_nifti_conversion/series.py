"""Read and write dicom files."""


import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from srmip.dicom_nifti_conversion.constants import (
    DICOM_FIELDS,
    SERIES_DEPENDENT_FIELDS,
    SLICE_DEPENDENT_FIELDS,
)
from srmip.utils import PathLike, format_digit_string

logger = logging.getLogger(__name__)


def get_series_dicom_files(dicom_series_directory_path: PathLike) -> Tuple[str]:
    """
    Get the list of dicom files of the series to be read.

    Read series ids first and then read the modalities. This is done in order to exclude
    RT Dose files.
    :param dicom_series_directory_path: Path of the directory containing the Dicom Series.
    :type dicom_series_directory_path: PathLike
    :return: Tuple of all full paths of the dicom slices (empty if no series are found
        in the directory).
    :type: Tuple[str]
    """
    series_ids = sitk.ImageSeriesReader().GetGDCMSeriesIDs(str(dicom_series_directory_path))
    for series_id in series_ids:
        dicom_series_files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
            str(dicom_series_directory_path), series_id
        )
        ds = pydicom.dcmread(dicom_series_files[0])
        if ds["Modality"].value != "RTDOSE":
            return dicom_series_files
    logger.warning("No Series can be found, make sure your restrictions are not too strong")
    return tuple()


def read_dicom_series(dicom_series_directory_path: PathLike) -> Tuple[sitk.Image, Dict[str, str]]:
    """
    Read Dicom series from file.

    :param dicom_series_directory_path: Path of the directory containing the Dicom Series.
    :type dicom_series_directory_path: PathLike
    :return: SimpleITK Image and metadata dictionary.
    :type: Tuple[sitk.Image, Dict[str, str]
    """
    dicom_series_files = get_series_dicom_files(dicom_series_directory_path)
    slices_number = len(dicom_series_files)
    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_series_reader.SetFileNames(dicom_series_files)
    dicom_series_reader.MetaDataDictionaryArrayUpdateOn()
    dicom_series_reader.LoadPrivateTagsOn()
    dicom_series = dicom_series_reader.Execute()
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
    return dicom_series, series_metadata


def write_dicom_series(
    image: sitk.Image, input_metadata: Dict[str, str], save_path: PathLike
) -> None:
    """
    Save the image as a Dicom series.

    :param image: image object to be saved.
    :type image: sitk.Image
    :param input_metadata: dictionary containing image metadata used for the creation
        of the Dicom header.
    :type input_metadata: Dict[str, str]
    :param save_path: Path where the image is saved. save_path must be a directory.
        If it does not already exist, a new dicrectory is created.
    :type save_path: PathLike
    """
    save_path = Path(save_path)
    # the following raises an error if save_path is an existing non-directory file
    save_path.mkdir(parents=True, exist_ok=True)
    series_writer = sitk.ImageFileWriter()
    series_writer.KeepOriginalImageUIDOn()

    instance_numbers = np.array(json.loads(input_metadata["slice_indexes"]))

    image_metadata = {}
    for dicom_tag in DICOM_FIELDS.values():
        if dicom_tag in input_metadata:
            image_metadata[dicom_tag] = input_metadata[dicom_tag]

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
