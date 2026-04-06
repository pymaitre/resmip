"""Abstraction for an annotation (Segmentation/Structure) on DICOM files."""

import re
from dataclasses import dataclass

import pydicom
import SimpleITK as sitk


@dataclass
class DicomAnnotation:
    """Structure read by the platipy wrapper."""

    name: str
    """Name of the annotation."""
    image: sitk.Image
    """Annotation mask as SimpleITK Image."""


def _duplicate_annotation_names(
    annotations: pydicom.Sequence, key: str, case_sensitive: bool = True
) -> bool:
    """Check if there are multiple annotations with the same name in the dataset.

    Args:
        annotations (pydicom.Sequence): Sequence of datasets containing
            information about annotations, including their name and index.
        key (str): DICOM key corresponding to the annotation name.
            RTSTRUCT uses "ROIName".
            SEG uses "SegmentLabel".
        case_sensitive (bool): When set to ``False``, ignore character case:
            e.g., ``"STRUCTURE" == "structure"``.

    Returns:
        bool: ``True`` if multiple annotations have the same name.
    """
    annotation_names = [
        a[key].value if case_sensitive else a[key].value.lower() for a in annotations
    ]
    return len(annotation_names) != len(set(annotation_names))


def _match_annotation_names(
    annotation_sequence: pydicom.Sequence,
    names_to_match: list[str],
    key: str,
    regex: bool,
    case_sensitive: bool = True,
) -> list[pydicom.Dataset]:
    """Find matching annotations in a DICOM sequence.

    Args:
        annotation_sequence (pydicom.Sequence): Sequence of datasets containing
            annotation names.
        names_to_match (list[str]): List of strings or regular expression patterns
            to search for within the sequence.
        key (str): DICOM key corresponding to the annotation name.
            RTSTRUCT uses "ROIName".
            SEG uses "SegmentLabel".
        regex (bool): If ``True``, treat the strings in ``names_to_match`` as regular
            expression patterns. If ``False``, perform an exact match.
        case_sensitive (bool): Whether the comparison should respect character
            casing.

    Returns:
        list[pydicom.Dataset]: List of dataset from the original sequence that met
            the matching criteria.
    """
    if case_sensitive:
        comparison_names = set(names_to_match)
    else:
        comparison_names = {n.lower() for n in names_to_match}

    matched_dicom_segments = []
    for dicom_seg_dataset in annotation_sequence:
        annotation_name_found = str(dicom_seg_dataset[key].value)
        annotation_name_found = (
            annotation_name_found if case_sensitive else annotation_name_found.lower()
        )
        if regex is True:
            flags = 0 if case_sensitive else re.IGNORECASE
            for annotation_name in comparison_names:
                if re.search(annotation_name, annotation_name_found, flags=flags):
                    matched_dicom_segments.append(dicom_seg_dataset)
                    break
        else:
            if annotation_name_found in comparison_names:
                matched_dicom_segments.append(dicom_seg_dataset)
    return matched_dicom_segments
