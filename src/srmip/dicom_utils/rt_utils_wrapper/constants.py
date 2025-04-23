"""Constants used for building DICOM RTst headers."""

from enum import Enum


class ROIGenerationAlgorithm(Enum):
    """
    ROI Generation Algorithm.

    For more information see here:
    https://dicom.innolitics.com/ciods/rt-structure-set/structure-set/30060020/30060036
    """

    null = 0
    automatic = 1
    semiautomatic = 2
    manual = 3


ROI_GENERATION_ALGORITHM = {
    ROIGenerationAlgorithm.null: "",
    ROIGenerationAlgorithm.automatic: "AUTOMATIC",
    ROIGenerationAlgorithm.semiautomatic: "SEMIAUTOMATIC",
    ROIGenerationAlgorithm.manual: "MANUAL",
}
"""
Type of algorithm used to generate ROI.

For more information see here:
https://dicom.innolitics.com/ciods/rt-structure-set/structure-set/30060020/30060036
"""
