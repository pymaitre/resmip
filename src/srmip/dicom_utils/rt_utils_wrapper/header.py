"""Functions used to read/write DICOM header information."""

import json
import re

import pydicom
from pydicom.uid import generate_uid


def add_leading_zero_to_header_value(value) -> str:
    """
    Add leading zero to numeric value in the header, if missing.

    :param value: Number read from the DICOM header.
    :type value: any
    :return: Number string with leading zeros added.
    :rtype: str
    """
    return re.sub(r"(\[| -?)\.", r"\g<1>0.", str(value))


def get_slice_positioning(dicom_slice: pydicom.Dataset) -> dict[str, tuple[float]]:
    """Get voxel spacing, origin and direction from a DICOM slice."""
    try:
        series_instance_uid = str(dicom_slice["SOPInstanceUID"].value)
    except KeyError as e:
        raise KeyError("Missing SOP Instance UID in at least one slice.") from e
    try:
        slice_z_spacing = str(dicom_slice["SliceThickness"].value)
    except KeyError as e:
        raise KeyError(f"Missing Slice Thickness in the slice {series_instance_uid}.") from e
    try:
        slice_xy_spacing = add_leading_zero_to_header_value(dicom_slice["PixelSpacing"].value)
    except KeyError as e:
        raise KeyError(f"Missing Pixel Spacing in the slice {series_instance_uid}.") from e
    try:
        slice_origin = json.loads(
            add_leading_zero_to_header_value(dicom_slice["ImagePositionPatient"].value)
        )
    except KeyError as e:
        raise KeyError(
            f"Missing Image Position (Patient) in the slice {series_instance_uid}."
        ) from e
    try:
        slice_direction = json.loads(str(dicom_slice["ImageOrientationPatient"].value))
    except KeyError as e:
        raise KeyError(
            f"Missing Image Orientation (Patient) in the slice {series_instance_uid}."
        ) from e
    slice_spacing = tuple(
        [float(value) for value in json.loads(slice_xy_spacing)] + [float(slice_z_spacing)]
    )
    return {
        "spacing": slice_spacing,
        "origin": slice_origin,
        "direction": slice_direction,
    }


def add_study_and_series_information(
    ds: pydicom.FileDataset, series_data: list[pydicom.Dataset], **kwargs
):
    """Add study information to the DICOM header."""
    reference_ds = series_data[0]  # All elements in series should have the same data
    series_description = kwargs.get("series_description", "")
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = series_description
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = (
        "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)
    )
