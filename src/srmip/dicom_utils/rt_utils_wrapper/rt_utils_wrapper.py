"""Wrapper module for rt_utils, used when converting nifti files to DICOM."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import pydicom
import SimpleITK as sitk
import skimage.measure

logger = logging.getLogger(__name__)


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


def create_contour(series_slice: pydicom.Dataset, contour_data: np.ndarray) -> pydicom.Dataset:
    """Create contours for a specific DICOM series slice."""
    contour_image = pydicom.Dataset()
    contour_image.ReferencedSOPClassUID = series_slice.SOPClassUID
    contour_image.ReferencedSOPInstanceUID = series_slice.SOPInstanceUID

    # Contour Image Sequence
    contour_image_sequence = pydicom.Sequence()
    contour_image_sequence.append(contour_image)

    contour = pydicom.Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = "CLOSED_PLANAR"  # TODO figure out how to get this value
    contour.NumberOfContourPoints = len(contour_data) // 3  # Each point has an x, y, and z value
    contour.ContourData = contour_data.tolist()

    return contour


def get_polygon_contours_from_slice_mask(slice_mask: np.ndarray) -> Tuple[np.ndarray]:
    """
    Convert the slice mask to a collection of polygon contours.

    :param slice_mask: Mask of the slice to be converted, of shape (x_dim, y_dim).
    :type slice_mask: np.ndarray
    :return: Tuple of polygon vertices (as x, y tuples) of the mask contour, of shape (n_points, 2).
        The length of the tuple is the number of polygons in the slice.
    :rtype: Tuple[np.ndarray]
    """
    polygons = tuple(
        np.flip(np.array(poly), axis=1) - 1
        for poly in skimage.measure.find_contours(np.pad(slice_mask, 1).astype(np.uint8))
    )
    # Add the first point after the last one, in order to fully
    # close the polygon, otherwise, in case of contours with holes,
    # a small slice on the xy plane would be skipped
    return tuple(np.concatenate((poly, [poly[0]])) for poly in polygons)


def get_contour_from_slice_mask(slice_mask: np.ndarray) -> np.ndarray:
    """
    Convert the slice mask to a polygon contour.

    :param slice_mask: Mask of the slice to be converted, of shape (x_dim, y_dim).
    :type slice_mask: np.ndarray
    :return: Polygon vertices (as x, y tuples) of the mask contour, of shape (n_points, 2).
    :rtype: np.ndarray
    """
    polygons = get_polygon_contours_from_slice_mask(slice_mask)
    # Connect each polygon with the first point of the first polygon,
    # in order to fully separate them
    corrected_polygons = tuple(
        np.concatenate((poly, [polygons[0][0]]), axis=0) for poly in polygons
    )
    return np.concatenate(corrected_polygons, axis=0)


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
    color: Union[str, List[int]] = None
    """Color of the RT structure."""
    description: str = ""
    """ROI description."""
    roi_generation_algorithm: Union[str, ROIGenerationAlgorithm] = ROIGenerationAlgorithm.null
    """
    Supported values:
        - ""
        - "AUTOMATIC"
        - "SEMIAUTOMATIC"
        - "MANUAL"
    """

    def __post_init__(self):
        """Cast ROI Generation Algorithm as string."""
        if isinstance(self.roi_generation_algorithm, ROIGenerationAlgorithm):
            self.roi_generation_algorithm = ROI_GENERATION_ALGORITHM[self.roi_generation_algorithm]

    def structure_set_roi(self) -> pydicom.Dataset:
        """Create the Structure Set ROI for the structure."""
        structure_set_roi = pydicom.Dataset()
        structure_set_roi.ROINumber = self.number
        structure_set_roi.ReferencedFrameOfReferenceUID = self.frame_of_reference_uid
        structure_set_roi.ROIName = self.name
        structure_set_roi.ROIDescription = self.description
        structure_set_roi.ROIGenerationAlgorithm = self.roi_generation_algorithm
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
        series_data: List[pydicom.Dataset],
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

    def validate_mask_array(self, mask: np.ndarray, series_data: List[pydicom.Dataset]) -> None:
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
