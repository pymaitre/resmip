"""Constant values present in Dicom headers."""

from pydicom.tag import BaseTag, Tag

SLICE_DEPENDENT_FIELDS = [
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
]
"""
Dicom tags that vary across slices in a series.

They need to be saved separately in the metadata file.
"""

SERIES_DEPENDENT_FIELDS = [
    # "SOPClassUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyID",
]
"""
Dicom tags that are unique to each Dicom series.

They must be generated every time a Dicom series is saved.
"""

DICOM_FIELDS = [
    # "SOPClassUID",
    "SOPInstanceUID",
    "StudyDate",
    "SeriesDate",
    "StudyTime",
    "Modality",
    "ReferringPhysicianName",
    "OperatorsName",
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "SliceThickness",
    "SpacingBetweenSlices",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyID",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "FrameOfReferenceUID",
    "SliceLocation",
    "PixelSpacing",
    # "RescaleIntercept",
    # "RescaleSlope",
    # "RescaleType",
]
"""DICOM fields saved in image metadata."""


def dicom_tag_to_string(tag: BaseTag) -> str:
    """Convert a DICOM tag to string.

    DICOM tags are hexadecimal integers.
    This function converts them to valid strings.
    As an example, ``Tag(0x12345678)`` becomes ``"1234|5678"``,
    ``Tag(0x120034)`` becomes ``"0012|0034"``.
    """
    return f"{tag.group:04x}|{tag.elem:04x}"


def string_tag_for_keyword(keyword: str) -> str | None:
    """Convert a keyword to a DICOM tag as string."""
    return dicom_tag_to_string(Tag(keyword))
