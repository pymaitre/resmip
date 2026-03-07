"""DICOM fields specific to image modalities."""

__all__ = ["PATIENT_RELATED_FIELDS", "STUDY_RELATED_FIELDS"]
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
