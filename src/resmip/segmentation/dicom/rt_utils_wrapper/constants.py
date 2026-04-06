"""Constants used for building DICOM RTst headers."""

from enum import Enum

__all__ = ["ROIGenerationAlgorithm"]


class ROIGenerationAlgorithm(Enum):
    """ROI Generation Algorithm.

    For more information see here:
    https://dicom.innolitics.com/ciods/rt-structure-set/structure-set/30060020/30060036
    """

    null = ""
    automatic = "AUTOMATIC"
    semiautomatic = "SEMIAUTOMATIC"
    manual = "MANUAL"
