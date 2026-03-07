"""ROI data used in the DICOM header."""

import logging
from dataclasses import dataclass

import numpy as np
import pydicom
import SimpleITK as sitk

from resmip.dicom_utils.rt_utils_wrapper.constants import (
    ROIGenerationAlgorithm,
)
from resmip.dicom_utils.rt_utils_wrapper.contours import (
    create_contour,
    get_contour_from_slice_mask,
)
from resmip.utils import format_dicom_code_string

logger = logging.getLogger(__name__)


@dataclass
class ROIData:
    """ROI data used for the DICOM header."""

    mask: sitk.Image
    """Binary mask image of the ROI."""
    number: int
    """Progressive ROI number (starting from 1)."""
    name: str
    """ROI name."""
    frame_of_reference_uid: str
    """Frame of reference of the referenced series."""
    color: str | list[int] = None
    """Color of the RT structure."""
    description: str = ""
    """ROI description."""
    roi_generation_algorithm: str | ROIGenerationAlgorithm = ROIGenerationAlgorithm.null
    """
    Supported values:
        - ""
        - "AUTOMATIC"
        - "SEMIAUTOMATIC"
        - "MANUAL"
    """

    def __post_init__(self):
        """Cast ROI Generation Algorithm as string."""
        if isinstance(self.roi_generation_algorithm, str):
            self.roi_generation_algorithm = ROIGenerationAlgorithm(
                format_dicom_code_string(self.roi_generation_algorithm)
            )

    def structure_set_roi(self) -> pydicom.Dataset:
        """Create the Structure Set ROI for the structure."""
        structure_set_roi = pydicom.Dataset()
        structure_set_roi.ROINumber = self.number
        structure_set_roi.ReferencedFrameOfReferenceUID = self.frame_of_reference_uid
        structure_set_roi.ROIName = self.name
        structure_set_roi.ROIDescription = self.description
        structure_set_roi.ROIGenerationAlgorithm = self.roi_generation_algorithm.value
        return structure_set_roi

    def rt_roi_observation(self) -> pydicom.Dataset:
        """Create the RT ROI Observation for the structure."""
        rtroi_observation = pydicom.Dataset()
        rtroi_observation.ObservationNumber = self.number
        rtroi_observation.ReferencedROINumber = self.number
        rtroi_observation.ROIObservationLabel = self.name
        rtroi_observation.RTROIInterpretedType = ""
        rtroi_observation.ROIInterpreter = ""
        return rtroi_observation

    def roi_contour_sequence(
        self,
        series_data: list[pydicom.Dataset],
    ) -> pydicom.Dataset:
        """Create the ROI Contour Sequence for the structure."""
        mask_array = sitk.GetArrayFromImage(self.mask) != 0
        mask_array = np.transpose(mask_array, (1, 2, 0))
        self.validate_mask_array(mask_array, series_data)
        roi_contour = pydicom.Dataset()
        roi_contour.ROIDisplayColor = self.color
        roi_contour.ReferencedROINumber = str(self.number)
        contour_sequence = pydicom.Sequence()
        for i in range(mask_array.shape[2]):
            roi_slice = mask_array[:, :, i]
            if np.all(roi_slice == 0):
                continue
            contour_points = get_contour_from_slice_mask(roi_slice)
            dicom_contour_points = np.concatenate(
                (contour_points, np.ones((contour_points.shape[0], 1)) * i), axis=1
            ).astype(float)
            dicom_contour_points = (
                (dicom_contour_points * self.mask.GetSpacing()) + self.mask.GetOrigin()
            ).ravel()
            dicom_contour = create_contour(series_data[i], dicom_contour_points)
            contour_sequence.append(dicom_contour)

        roi_contour.ContourSequence = contour_sequence
        return roi_contour

    def validate_mask_array(self, mask: np.ndarray, series_data: list[pydicom.Dataset]) -> None:
        """Check if the mask has correct type and shape."""
        if mask.dtype != bool:
            raise TypeError(f"Mask data type must be boolean. Got {mask.dtype}")

        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3 dimensional. Got {mask.ndim}")

        if len(series_data) != np.shape(mask)[2]:
            raise ValueError(
                "Mask must have the save number of layers (In the 3rd dimension) as input series. "
                f"Expected {len(series_data)}, got {np.shape(mask)[2]}"
            )

        if mask.sum() == 0:
            logger.info("ROI mask is empty")
