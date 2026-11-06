"""Shared utilities for building DICOM datasets for RTSTRUCT and SEG."""

from pydicom import Dataset, FileMetaDataset
from pydicom._uid_dict import UID_dictionary
from pydicom.uid import UID, generate_uid

from resmip.dicom_utils.constants import _RESMIP_IMPLEMENTATION_CLASS_UID

_INVERSE_UID_DICTIONARY = {v[0]: k for k, v in UID_dictionary.items()}
"""Reverse mapping from UID name to UID string, derived from pydicom's UID dictionary."""

_OPTIONAL_PATIENT_KEYWORDS = [
    "PatientName",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
]
"""Patient-level tags copied only when present in the reference dataset."""

_MANDATORY_PATIENT_KEYWORDS = [
    "PatientID",
]
"""Patient-level tags always copied from the reference dataset."""

_OPTIONAL_STUDY_KEYWORDS = [
    "StudyDate",
    "StudyTime",
    "StudyID",
    "StudyDescription",
]
"""Study-level tags copied only when present in the reference dataset."""

_MANDATORY_STUDY_KEYWORDS = [
    "StudyInstanceUID",
]
"""Study-level tags always copied from the reference dataset."""


def _inverse_uid_lookup(name: str):
    """Look up a DICOM UID string by its human-readable name.

    Args:
        name (str): Human-readable UID name as defined in the DICOM
            standard, e.g. ``"Segmentation Storage"`` or
            ``"RT Structure Set Storage"``.

    Returns:
        UID: The corresponding DICOM UID.
    """
    return UID(_INVERSE_UID_DICTIONARY[name])


def _setup_file_meta(storage_type: str, *, transfer_syntax_uid: UID):
    """Create a DICOM File Meta Information dataset for a given storage type.

    Generates a new ``MediaStorageSOPInstanceUID`` on every call.
    The transfer syntax is keyword-only and has no default because the right choice
    depends on the elements of the IOD being written: each writer passes the default
    defined in its own module instead of inheriting a shared one.

    Args:
        storage_type (str): Human-readable name of the SOP storage class,
            e.g. ``"Segmentation Storage"`` or ``"RT Structure Set Storage"``.
        transfer_syntax_uid (UID): Transfer syntax used to encode the data set.

    Returns:
        FileMetaDataset: Populated file meta dataset ready to attach to a
            ``FileDataset``.
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = _inverse_uid_lookup(storage_type)
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = transfer_syntax_uid
    file_meta.ImplementationClassUID = _RESMIP_IMPLEMENTATION_CLASS_UID
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
