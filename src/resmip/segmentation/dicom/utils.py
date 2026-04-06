""""""

from pydicom import Dataset, FileMetaDataset
from pydicom._uid_dict import UID_dictionary
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

_INVERSE_UID_DICTIONARY = {v[0]: k for k, v in UID_dictionary.items()}

_OPTIONAL_PATIENT_KEYWORDS = [
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
]

_MANDATORY_PATIENT_KEYWORDS = [
    "PatientID",
]

_OPTIONAL_STUDY_KEYWORDS = [
    "StudyDate",
    "StudyTime",
    "StudyID",
    "StudyDescription",
]

_MANDATORY_STUDY_KEYWORDS = [
    "StudyInstanceUID",
]


def _inverse_uid_lookup(name: str):
    return UID(_INVERSE_UID_DICTIONARY[name])


def _setup_file_meta(storage_type: str):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = _inverse_uid_lookup(storage_type)
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.276.0.7230010.3.0.3.6.1"
    file_meta.ImplementationVersionName = "OFFIS_DCMTK_361"
    return file_meta


def _copy_patient_and_study_information(ds: Dataset, reference_ds: Dataset) -> None:
    """Copy patient and study tags from a reference DICOM dataset.

    Optional tags are copied only if present in the reference.
    Mandatory tags are always copied and will raise AttributeError
    if missing from the reference.

    Args:
        ds (Dataset): Target dataset to populate.
        reference_ds (Dataset): Source dataset to read tags from,
            typically the first slice of the reference series.
    """
    for keyword in _OPTIONAL_PATIENT_KEYWORDS + _OPTIONAL_STUDY_KEYWORDS:
        if keyword in reference_ds:
            setattr(ds, keyword, getattr(reference_ds, keyword))
    for keyword in _MANDATORY_PATIENT_KEYWORDS + _MANDATORY_STUDY_KEYWORDS:
        setattr(ds, keyword, getattr(reference_ds, keyword))
