"""Wrapper module for rt_utils, used when converting nifti files to DICOM."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pydicom
import rt_utils
import rt_utils.ds_helper
import rt_utils.image_helper
import SimpleITK as sitk
import skimage.measure
from pydicom.uid import generate_uid

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


def add_leading_zero_to_header_value(value) -> str:
    """
    Add leading zero to numeric value in the header, if missing.

    :param value: Number read from the DICOM header.
    :type value: any
    :return: Number string with leading zeros added.
    :rtype: str
    """
    return re.sub(r"(\[| -?)\.", r"\g<1>0.", str(value))


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
        slice_xy_spacing = add_leading_zero_to_header_value(dicom_slice["PixelSpacing"].value)
    except KeyError as e:
        raise KeyError(f"Missing Pixel Spacing in the slice {series_instance_uid}.") from e
    try:
        slice_origin = json.loads(
            add_leading_zero_to_header_value(dicom_slice["ImagePositionPatient"].value)
        )
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


def add_study_and_series_information(
    ds: pydicom.FileDataset, series_data: List[pydicom.Dataset], **kwargs
):
    """Add study information to the DICOM header."""
    reference_ds = series_data[0]  # All elements in series should have the same data
    series_description = kwargs.get("series_description", "")
    ds.StudyDate = reference_ds.StudyDate
    ds.SeriesDate = getattr(reference_ds, "SeriesDate", "")
    ds.StudyTime = reference_ds.StudyTime
    ds.SeriesTime = getattr(reference_ds, "SeriesTime", "")
    ds.StudyDescription = getattr(reference_ds, "StudyDescription", "")
    ds.SeriesDescription = series_description
    ds.StudyInstanceUID = reference_ds.StudyInstanceUID
    ds.SeriesInstanceUID = generate_uid()  # TODO: find out if random generation is ok
    ds.StudyID = reference_ds.StudyID
    ds.SeriesNumber = (
        "1"  # TODO: find out if we can just use 1 (Should be fine since its a new series)
    )


def create_rtstruct_dataset(series_data: List[pydicom.Dataset], **kwargs) -> pydicom.FileDataset:
    """Create the DICOM header template for the RT Structure Set."""
    ds = rt_utils.ds_helper.generate_base_dataset()
    add_study_and_series_information(ds, series_data, **kwargs)
    rt_utils.ds_helper.add_patient_information(ds, series_data)
    rt_utils.ds_helper.add_refd_frame_of_ref_sequence(ds, series_data)
    return ds


class RTStruct:
    """Wrapper class of rt_utils.RTStruct."""

    def __init__(self, series_data: List[pydicom.Dataset], ds: pydicom.FileDataset):
        """Instantiate the RT structure set file builder."""
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID

    @classmethod
    def create_new(cls, dicom_series_path: str, **kwargs) -> RTStruct:
        """Create new RTStruct given the path of the referenced DICOM series."""
        series_data = rt_utils.image_helper.load_sorted_image_series(dicom_series_path)
        ds = create_rtstruct_dataset(series_data, **kwargs)
        return cls(series_data, ds)

    def add_roi(
        self,
        mask: sitk.Image,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = "",
        roi_generation_algorithm: Union[str, ROIGenerationAlgorithm] = ROIGenerationAlgorithm.null,
    ):
        """Add contour to the RT structure set file."""
        self.validate_mask(mask)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_data = ROIData(
            mask,
            roi_number,
            name,
            self.frame_of_reference_uid,
            color,
            description,
            roi_generation_algorithm,
        )

        self.ds.ROIContourSequence.append(roi_data.roi_contour_sequence(self.series_data))
        self.ds.StructureSetROISequence.append(roi_data.structure_set_roi())
        self.ds.RTROIObservationsSequence.append(roi_data.rt_roi_observation())

    def validate_mask(self, mask: sitk.Image) -> None:
        """Check if the mask has correct origin, spacing and orientation."""
        reference_spacing = None
        reference_origin = None
        reference_direction = None
        z_values = []
        for dicom_slice in self.series_data:
            slice_positioning = get_slice_positioning(dicom_slice)
            if reference_spacing is None:
                reference_spacing = slice_positioning["spacing"]
            if reference_origin is None:
                reference_origin = slice_positioning["origin"]
            if reference_direction is None:
                reference_direction = slice_positioning["direction"]
            z_values.append(slice_positioning["origin"][2])
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
        z_values = np.array(z_values).round(decimals=2)
        if len(z_values) > 1:
            z_spacing = (z_values.max() - z_values.min()) / (len(z_values) - 1)
        else:
            z_spacing = 1

        reference_spacing = (reference_spacing[:2]) + (z_spacing,)

        # if mask.GetSpacing() != reference_spacing:
        if not np.allclose(mask.GetSpacing(), reference_spacing):
            raise ValueError(
                f"The mask spacing ({mask.GetSpacing()}) is different "
                f"than the reference series spacing ({reference_spacing})."
            )
        # if mask.GetOrigin() != tuple(reference_origin):
        if not np.allclose(mask.GetOrigin(), reference_origin):
            raise ValueError(
                f"The mask origin ({mask.GetOrigin()}) is different "
                f"than the reference series origin ({reference_origin})."
            )
        if mask.GetDirection()[:-3] != tuple(reference_direction):
            raise ValueError(
                f"The mask origin ({mask.GetDirection()[:-3]}) is different "
                f"than the reference series origin ({reference_direction})."
            )

    def save(self, file_path: Path) -> None:
        """Saves the RTStruct with the specified name / location."""
        logger.info("Writing file to %s", file_path)
        self.ds.save_as(file_path)
