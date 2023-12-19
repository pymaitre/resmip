"""Constant values present in Dicom headers."""
from bidict import bidict

SLICE_DEPENDENT_FIELDS = [
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
]

SERIES_DEPENDENT_FIELDS = [
    # "SOPClassUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "StudyID",
]

# TODO: Add relevant tags (e.g.: clinician name)
DICOM_FIELDS = bidict(
    {
        # "SOPClassUID": "0008|0016",
        "SOPInstanceUID": "0008|0018",
        "StudyDate": "0008|0020",
        "SeriesDate": "0008|0021",
        "StudyTime": "0008|0030",
        "Modality": "0008|0060",
        "ReferringPhysicianName": "0008|0090",
        "PatientName": "0010|0010",
        "PatientID": "0010|0020",
        "PatientBirthDate": "0010|0030",
        "PatientSex": "0010|0040",
        # "0018|0050": "SliceThickness",
        # "0018|0088": "SpacingBetweenSlices",
        "StudyInstanceUID": "0020|000D",
        "SeriesInstanceUID": "0020|000E",
        "StudyID": "0020|0010",
        "InstanceNumber": "0020|0013",
        "ImagePositionPatient": "0020|0032",
        "ImageOrientationPatient": "0020|0037",
        "FrameOfReferenceUID": "0020|0052",
        "SliceLocation": "0020|1041",
        # "RescaleIntercept": "0028|1052",
        # "RescaleSlope": "0028|1053",
        # "RescaleType": "0028|1054",
    }
)
