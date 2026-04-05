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
    seg = "SEG"
    """Segmentation."""


SERIES_MODALITIES = [
    DicomModality.ct,
    DicomModality.mr,
    DicomModality.pt,
]
"""Modalities used in series images."""
