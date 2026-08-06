"""Constant values present in Dicom headers."""

import hashlib

from pydicom.tag import BaseTag, Tag
from pydicom.uid import UID

__all__ = ["string_tag_for_keyword"]

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
]
"""
Dicom tags that are unique to each Dicom series.

They must be generated every time a Dicom series is saved.
"""

DICOM_FIELDS = [
    "ImageType",
    # "SOPClassUID",
    "SOPInstanceUID",
    "StudyDate",
    "SeriesDate",
    "StudyTime",
    "Modality",
    "Manufacturer",
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
    "PatientPosition",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyID",
    "AcquisitionNumber",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "FrameOfReferenceUID",
    "PositionReferenceIndicator",
    "SliceLocation",
    "PixelSpacing",
    # "RescaleIntercept",
    # "RescaleSlope",
    # "RescaleType",
]
"""DICOM fields saved in image metadata."""

SERIES_TYPE_1_ATTRIBUTES = {
    "ImageType": r"DERIVED\SECONDARY",
    "AcquisitionNumber": 1,
}
"""DICOM attributes that are defined as Type 1 for Series.

Type 1 attributes are required and cannot be empty.
This dictionary provides default values used by the library.
"""

SERIES_TYPE_2_ATTRIBUTES = [
    "PositionReferenceIndicator",
    "Manufacturer",
    "PatientPosition",
]
"""DICOM attributes that are defined as Type 2 for Series.

Type 2 attributes are required and can be empty.
"""

_RESMIP_IMPLEMENTATION_CLASS_UID = UID(
    "2.25." + str(int(hashlib.md5(b"resmip").hexdigest(), 16))[:39]
)
"""Stable DICOM Implementation Class UID for resmip.

Derived deterministically from the package name using the 2.25 UUID-based
UID root, which is reserved for this purpose in the DICOM standard and does
not require registration. Used in File Meta Information datasets to identify
resmip as the creating implementation.
"""


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
