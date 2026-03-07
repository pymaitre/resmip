"""Image metadata."""

from enum import Enum

__all__ = ["DicomModality", "SERIES_MODALITIES"]


class DicomModality(Enum):
    """DICOM modalities supported by resmip."""

    ct = "CT"
    """Computed Tomography."""
    mr = "MR"
    """Magnetic Resonance."""
    pt = "PT"
    """Positron emission tomography (PET)."""
    rtdose = "RTDOSE"
    """Radiotherapy Dose."""
    rtstruct = "RTSTRUCT"
    """Radiotherapy Structure Set."""


SERIES_MODALITIES = [
    DicomModality.ct,
    DicomModality.mr,
    DicomModality.pt,
]
"""Modalities used in series images."""

PATIENT_RELATED_FIELDS = [
    "PatientName",
    "IssuerOfPatientID",
    "IssuerOfPatientID",
    "TypeOfPatientID",
    "IssuerOfPatientIDQualifiersSequence",
    "SourcePatientGroupIdentificationSequence",
    "GroupOfPatientsIdentificationSequence",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientBirthDateInAlternativeCalendar",
    "PatientDeathDateInAlternativeCalendar",
    "PatientAlternativeCalendar",
    "PatientSex",
    "QualityControlSubject",
    "StrainDescription",
    "StrainNomenclature",
    "StrainStockSequence",
    "StrainAdditionalInformation",
    "StrainCodeSequence",
    "GeneticModificationsSequence",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
]
"""DICOM fields specifically related to patient information."""

STUDY_RELATED_FIELDS = [
    "StudyDate",
    "StudyTime",
    "AccessionNumber",
    "ReferringPhysicianName",
    "ConsultingPhysicianName",
    "StudyDescription",
    "PhysiciansOfRecord",
    "NameOfPhysiciansReadingStudy",
    "StudyID",
    "RequestingService",
]
"""DICOM fields specifically related to study information.

In particular, they refer to the "General Study" module.
"""
