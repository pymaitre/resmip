"""Wrapper module for rt_utils, used when converting nifti files to DICOM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pydicom
import rt_utils
import rt_utils.ds_helper
import rt_utils.image_helper
import rt_utils.utils
import SimpleITK as sitk

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
    mask: sitk.Image,
    series_data: List[pydicom.Dataset],
    color,
    roi_number: int,
) -> pydicom.Dataset:
    """Generate contour sequence for a specific structure."""
    mask_array = sitk.GetArrayFromImage(mask) != 0
    mask_array = np.transpose(mask_array, (1, 2, 0))
    roi_contour = pydicom.Dataset()
    roi_contour.ROIDisplayColor = color
    roi_contour.ReferencedROINumber = str(roi_number)
    contour_sequence = pydicom.Sequence()
    for i in range(mask_array.shape[2]):
        roi_slice = mask_array[:, :, i]
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
        dicom_contour_points = (
            (dicom_contour_points * mask.GetSpacing()) + mask.GetOrigin()
        ).ravel()
        dicom_contour = create_contour(series_data[i], dicom_contour_points)
        contour_sequence.append(dicom_contour)

    roi_contour.ContourSequence = contour_sequence
    return roi_contour


def get_slice_positioning(dicom_slice: pydicom.Dataset) -> Dict[str, Tuple[float]]:
    """Get voxel spacing, origin and direction from a DICOM slice."""
    try:
        series_instance_uid = str(dicom_slice["SOPInstanceUID"].value)
    except KeyError as e:
        raise KeyError("Missing SOP Instance UID in at least one slice.") from e
    try:
        slice_z_spacing = str(dicom_slice["SliceThickness"].value)
    except KeyError as e:
        raise KeyError(f"Missing Slice Thickness in the slice {series_instance_uid}.") from e
    try:
        slice_xy_spacing = str(dicom_slice["PixelSpacing"].value)
    except KeyError as e:
        raise KeyError(f"Missing Pixel Spacing in the slice {series_instance_uid}.") from e
    try:
        slice_origin = json.loads(str(dicom_slice["ImagePositionPatient"].value))
    except KeyError as e:
        raise KeyError(
            f"Missing Image Position (Patient) in the slice {series_instance_uid}."
        ) from e
    try:
        slice_direction = json.loads(str(dicom_slice["ImageOrientationPatient"].value))
    except KeyError as e:
        raise KeyError(
            f"Missing Image Orientation (Patient) in the slice {series_instance_uid}."
        ) from e
    slice_spacing = tuple(
        [float(value) for value in json.loads(slice_xy_spacing)] + [float(slice_z_spacing)]
    )
    return {
        "spacing": slice_spacing,
        "origin": slice_origin,
        "direction": slice_direction,
    }


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
        mask: sitk.Image,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = "",
        use_pin_hole: bool = False,
        approximate_contours: bool = True,
        roi_generation_algorithm: Union[str, int] = 0,
    ):
        # TODO test if name already exists
        mask_array = sitk.GetArrayFromImage(mask) != 0
        mask_array = np.transpose(mask_array, (1, 2, 0))
        self.validate_mask_array(mask_array)
        self.validate_mask(mask)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = rt_utils.utils.ROIData(
            "mask",
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
            create_roi_contour(mask, self.series_data, color, roi_number)
        )
        self.ds.StructureSetROISequence.append(
            rt_utils.ds_helper.create_structure_set_roi(roi_data)
        )
        self.ds.RTROIObservationsSequence.append(
            rt_utils.ds_helper.create_rtroi_observation(roi_data)
        )

    def validate_mask_array(self, mask: np.ndarray) -> None:
        """Check if the mask has correct type and shape."""
        if mask.dtype != bool:
            raise TypeError(f"Mask data type must be boolean. Got {mask.dtype}")

        if mask.ndim != 3:
            raise ValueError(f"Mask must be 3 dimensional. Got {mask.ndim}")

        if len(self.series_data) != np.shape(mask)[2]:
            raise ValueError(
                "Mask must have the save number of layers (In the 3rd dimension) as input series. "
                f"Expected {len(self.series_data)}, got {np.shape(mask)[2]}"
            )

        if mask.sum() == 0:
            logger.info("ROI mask is empty")

    def validate_mask(self, mask: sitk.Image) -> None:
        """Check if the mask has correct origin, spacing and orientation."""
        reference_spacing = None
        reference_origin = None
        reference_direction = None
        for dicom_slice in self.series_data:
            slice_positioning = get_slice_positioning(dicom_slice)
            if reference_spacing is None:
                reference_spacing = slice_positioning["spacing"]
            if reference_origin is None:
                reference_origin = slice_positioning["origin"]
            if reference_direction is None:
                reference_direction = slice_positioning["direction"]
            reference_origin[2] = min(reference_origin[2], slice_positioning["origin"][2])
            assert (
                reference_spacing == slice_positioning["spacing"]
            ), "Nonuniform xy/slice spacing in the referenced series."
            assert (
                reference_origin[:2] == slice_positioning["origin"][:2]
            ), "Nonuniform image origin in the referenced series."
            assert (
                reference_direction == slice_positioning["direction"]
            ), "Nonuniform image direction in the referenced series."

        if mask.GetSpacing() != reference_spacing:
            raise ValueError(
                f"The mask spacing ({mask.GetSpacing()}) is different than the reference series spacing ({reference_spacing})."
            )
        if mask.GetOrigin() != tuple(reference_origin):
            raise ValueError(
                f"The mask origin ({mask.GetOrigin()}) is different than the reference series origin ({reference_origin})."
            )
        if mask.GetDirection()[:-3] != tuple(reference_direction):
            raise ValueError(
                f"The mask origin ({mask.GetDirection()[:-3]}) is different than the reference series origin ({reference_direction})."
            )

    def save(self, file_path: Path) -> None:
        """Saves the RTStruct with the specified name / location."""
        logger.info("Writing file to %s", file_path)
        self.ds.save_as(file_path)
