"""Wrapper module for rt_utils, used when converting nifti files to DICOM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import pydicom
import rt_utils
import rt_utils.ds_helper
import rt_utils.image_helper
import rt_utils.utils

logger = logging.getLogger(__name__)


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


def create_roi_contour(
    mask: np.ndarray,
    series_data: List[pydicom.Dataset],
    color,
    roi_number: int,
    spacing: Tuple[float],
    origin: Tuple[float],
) -> pydicom.Dataset:
    """Generate contour sequence for a specific structure."""
    roi_contour = pydicom.Dataset()
    roi_contour.ROIDisplayColor = color
    roi_contour.ReferencedROINumber = str(roi_number)
    contour_sequence = pydicom.Sequence()
    for i in range(mask.shape[2]):
        roi_slice = mask[:, :, i]
        if np.all(roi_slice == 0):
            continue
        contour_points = cv2.findContours(
            roi_slice.astype(np.uint8),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE,
        )[0][0][:, 0, :]
        dicom_contour_points = np.concatenate(
            (contour_points, np.ones((contour_points.shape[0], 1)) * i), axis=1
        ).astype(float)
        dicom_contour_points = ((dicom_contour_points * spacing) + origin).ravel()
        dicom_contour = create_contour(series_data[i], dicom_contour_points)
        contour_sequence.append(dicom_contour)

    roi_contour.ContourSequence = contour_sequence
    return roi_contour


class RTStruct(rt_utils.RTStruct):
    """Wrapper class of rt_utils.RTStruct."""

    @classmethod
    def create_new(cls, dicom_series_path: str) -> RTStruct:
        """Create new RTStruct given the path of the referenced DICOM series."""
        series_data = rt_utils.image_helper.load_sorted_image_series(dicom_series_path)
        ds = rt_utils.ds_helper.create_rtstruct_dataset(series_data)
        return cls(series_data, ds)

    def add_roi(
        self,
        mask: np.ndarray,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = "",
        use_pin_hole: bool = False,
        approximate_contours: bool = True,
        roi_generation_algorithm: Union[str, int] = 0,
        spacing: Tuple[float] = None,
        origin: Tuple[float] = None,
    ):  # pylint: disable=too-many-arguments
        if spacing is None or origin is None:
            raise ValueError("Both spacing and origin values must be specified.")
        # TODO test if name already exists
        self.validate_mask(mask)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = rt_utils.utils.ROIData(
            mask,
            color,
            roi_number,
            name,
            self.frame_of_reference_uid,
            description,
            use_pin_hole,
            approximate_contours,
            roi_generation_algorithm,
        )

        self.ds.ROIContourSequence.append(
            create_roi_contour(
                mask, self.series_data, color, roi_number, spacing=spacing, origin=origin
            )
        )
        self.ds.StructureSetROISequence.append(
            rt_utils.ds_helper.create_structure_set_roi(roi_data)
        )
        self.ds.RTROIObservationsSequence.append(
            rt_utils.ds_helper.create_rtroi_observation(roi_data)
        )

    def save(self, file_path: Path) -> None:
        """Saves the RTStruct with the specified name / location."""
        logger.info("Writing file to %s", file_path)
        self.ds.save_as(file_path)
