"""Read and write dicom files."""

import json
import logging
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from resmip.dicom_utils.constants import (
    DICOM_FIELDS,
    SERIES_DEPENDENT_FIELDS,
    SLICE_DEPENDENT_FIELDS,
    string_tag_for_keyword,
)
from resmip.image.metadata import SERIES_MODALITIES
from resmip.utils import PathLike, format_digit_string

logger = logging.getLogger(__name__)


def get_series_dicom_files(dicom_series_directory_path: PathLike) -> tuple[Path, ...]:
    """Get the list of dicom files of the series to be read.

    Read series ids first and then read the modalities. This is done in order to exclude
    RT Dose files and similar. Only ``SERIES_MODALITIES`` are supported.

    Args:
        dicom_series_directory_path (PathLike): Path of the directory containing the Dicom Series.

    Returns:
        tuple[Path, ...]: Tuple of all full paths of the dicom slices (empty if no series are found
        in the directory). The elements are casted from `str` to `Path` in order to return
        correct paths on Windows.
    """
    series_ids = sitk.ImageSeriesReader().GetGDCMSeriesIDs(str(dicom_series_directory_path))
    for series_id in series_ids:
        dicom_series_files = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(
            str(dicom_series_directory_path), series_id
        )
        ds = pydicom.dcmread(dicom_series_files[0])
        if ds["Modality"].value in (modality.value for modality in SERIES_MODALITIES):
            return tuple(Path(file_path) for file_path in dicom_series_files)
    logger.warning("No Series can be found, make sure your restrictions are not too strong")
    return tuple()


def get_spacing_from_dicom_header(
    dicom_series_reader: sitk.ImageSeriesReader, slices_number: int
) -> tuple[float, float, float]:
    """Read correctly-rounded voxel spacing from the DICOM header.

    If the series has only one slice, the z spacing is set to 1 mm.

    Args:
        dicom_series_reader (sitk.ImageSeriesReader): ITK DICOM reader for series.
        slices_number (int): Number of slices for the DICOM series.

    Returns:
        tuple[float, float, float]: (x, y, z) voxel spacing in mm.
    """
    xy_spacing = None
    z_values = []
    for i in range(slices_number):
        slice_xy_spacing = [
            float(spacing)
            for spacing in dicom_series_reader.GetMetaData(
                i, string_tag_for_keyword("PixelSpacing")
            ).split("\\")
        ]

        slice_z = [
            float(z)
            for z in dicom_series_reader.GetMetaData(
                i, string_tag_for_keyword("ImagePositionPatient")
            ).split("\\")
        ][2]
        z_values.append(float(slice_z))
        if xy_spacing is None:
            xy_spacing = slice_xy_spacing
        assert xy_spacing == slice_xy_spacing, "Nonuniform xy spacing detected"
    z_values = np.array(z_values).round(decimals=2)
    if len(z_values) > 1:
        slice_z_spacing = (z_values.max() - z_values.min()) / (len(z_values) - 1)
    else:
        logger.warning("Only 1 slice detected. Setting z voxel spacing to 1 mm.")
        slice_z_spacing = 1
    return tuple(slice_xy_spacing + [slice_z_spacing])


def read(dicom_series_directory_path: PathLike) -> tuple[sitk.Image, dict[str, str]]:
    """Read Dicom series from file.

    Non-unicode characters in the dicom header are escaped into unicode sequences.

    Args:
        dicom_series_directory_path (PathLike): Path of the directory containing the Dicom Series.

    Returns:
        tuple[sitk.Image, dict[str, str]]: SimpleITK Image and metadata dictionary.
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
            series_metadata.pop(string_tag_for_keyword(slice_dependent_field))
        except KeyError:
            # if the key is not present in the image, we must find a way
            # to keep track of this information
            pass

    instance_numbers = np.zeros(slices_number, dtype=int)
    image_position_patients = np.zeros((slices_number, 3), dtype=float)
    for i in range(slices_number):
        instance_number = dicom_series_reader.GetMetaData(
            i, string_tag_for_keyword("InstanceNumber")
        )
        image_position_patient = dicom_series_reader.GetMetaData(
            i, string_tag_for_keyword("ImagePositionPatient")
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
        dicom_series.SetMetaData(key, value.encode("unicode_escape").decode())

    dicom_spacing = get_spacing_from_dicom_header(dicom_series_reader, slices_number)
    dicom_series.SetSpacing(dicom_spacing)

    return dicom_series, series_metadata


def write(image: sitk.Image, input_metadata: dict[str, str], save_path: PathLike) -> None:
    """Save the image as a Dicom series.

    Args:
        image (sitk.Image): image object to be saved.
        input_metadata (dict[str, str]): dictionary containing image metadata used for the creation
            of the Dicom header.
        save_path (PathLike): Path where the image is saved. save_path must be a directory.
            If it does not already exist, a new dicrectory is created.
    """
    save_path = Path(save_path)
    # the following raises an error if save_path is an existing non-directory file
    save_path.mkdir(parents=True, exist_ok=True)
    series_writer = sitk.ImageFileWriter()
    series_writer.KeepOriginalImageUIDOn()

    image_metadata = {}
    for dicom_keyword in DICOM_FIELDS:
        dicom_tag = string_tag_for_keyword(dicom_keyword)
        if dicom_tag in input_metadata:
            image_metadata[dicom_tag] = input_metadata[dicom_tag]

    for series_dependent_field in SERIES_DEPENDENT_FIELDS:
        image_metadata[string_tag_for_keyword(series_dependent_field)] = pydicom.uid.generate_uid()

    if hasattr(image, "modality"):
        image_metadata[string_tag_for_keyword("Modality")] = image.modality
    # Do we want to round it back to the value of the Dicom or do we want
    # to keep the value computed by SimpleITK? The pixel grid on the Dicom file
    # is identical to the generated one.
    # Maybe it's safer to round it.
    # image_metadata["0018|0050"] = str(image.GetSpacing()[2])
    rounded_z_spacing = float(f"{image.GetSpacing()[2]:.3e}")
    image_metadata[string_tag_for_keyword("SliceThickness")] = str(rounded_z_spacing)
    image_metadata[string_tag_for_keyword("SpacingBetweenSlices")] = str(rounded_z_spacing)

    direction = image.GetDirection()
    image_metadata[string_tag_for_keyword("ImageOrientationPatient")] = "\\".join(
        map(
            str,
            (
                *direction[0::3],
                *direction[1::3],
            ),
        )
    )
    image_metadata[string_tag_for_keyword("PixelSpacing")] = "\\".join(
        map(str, image.GetSpacing()[:2])
    )

    for i in range(image.GetDepth()):
        image_slice = image[:, :, i]

        slice_metadata = image_metadata.copy()
        slice_metadata[string_tag_for_keyword("InstanceNumber")] = i

        for key, value in slice_metadata.items():
            image_slice.SetMetaData(key, str(value))

        # maybe these are not needed
        # image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
        # image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

        image_slice.SetMetaData(
            string_tag_for_keyword("ImagePositionPatient"),
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
