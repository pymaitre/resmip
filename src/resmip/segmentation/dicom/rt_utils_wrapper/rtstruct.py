"""Dicom RT Structure Set with header data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from resmip.dicom_utils.utils import _set_content_datetime

from ..utils import _copy_patient_and_study_information
from .constants import ROIGenerationAlgorithm
from .ds_helper import (
    add_refd_frame_of_ref_sequence,
    generate_base_dataset,
)
from .header import (
    add_study_and_series_information,
    get_slice_positioning,
)
from .image_helper import load_sorted_image_series
from .roidata import ROIData

logger = logging.getLogger(__name__)


def create_rtstruct_dataset(series_data: list[pydicom.Dataset], **kwargs) -> pydicom.FileDataset:
    """Create the DICOM header template for the RT Structure Set."""
    ds = generate_base_dataset()
    add_study_and_series_information(ds, series_data, **kwargs)
    _copy_patient_and_study_information(ds, series_data[0])
    add_refd_frame_of_ref_sequence(ds, series_data)
    _set_content_datetime(ds)
    return ds


class RTStruct:
    """Wrapper class of rt_utils.RTStruct."""

    def __init__(self, series_data: list[pydicom.Dataset], ds: pydicom.FileDataset):
        """Instantiate the RT structure set file builder."""
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID

    @classmethod
    def create_new(cls, dicom_series_path: str, **kwargs) -> RTStruct:
        """Create new RTStruct given the path of the referenced DICOM series."""
        series_data = load_sorted_image_series(dicom_series_path)
        ds = create_rtstruct_dataset(series_data, **kwargs)
        return cls(series_data, ds)

    def add_roi(
        self,
        *,
        mask: sitk.Image,
        color: str | list[int] = None,
        name: str = None,
        description: str = "",
        roi_generation_algorithm: str | ROIGenerationAlgorithm = ROIGenerationAlgorithm.null,
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

        if not np.allclose(mask.GetSpacing(), reference_spacing):
            raise ValueError(
                f"The mask spacing ({mask.GetSpacing()}) is different "
                f"than the reference series spacing ({reference_spacing})."
            )
        if not np.allclose(mask.GetOrigin(), reference_origin):
            raise ValueError(
                f"The mask origin ({mask.GetOrigin()}) is different "
                f"than the reference series origin ({reference_origin})."
            )
        if not np.allclose(mask.GetDirection()[:-3], reference_direction):
            raise ValueError(
                f"The mask direction ({mask.GetDirection()[:-3]}) is different "
                f"than the reference series direction ({reference_direction})."
            )

    def save(self, file_path: Path) -> None:
        """Saves the RTStruct with the specified name / location."""
        logger.info("Writing file to %s", file_path)
        self.ds.save_as(file_path, implicit_vr=False, little_endian=True)
