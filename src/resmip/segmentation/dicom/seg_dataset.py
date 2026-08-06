"""Abstraction of the ``pydicom.Dataset`` used for DICOM SEG."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydicom import Dataset, FileDataset, Sequence, dcmread, dcmwrite
from pydicom.tag import Tag
from pydicom.uid import generate_uid

from resmip._version import __version__
from resmip.dicom_utils.rt_utils_wrapper.image_helper import load_sorted_image_series
from resmip.dicom_utils.series import get_series_dicom_files
from resmip.dicom_utils.utils import _set_content_datetime
from resmip.segmentation.utils import SegmentationType

from .utils import (
    _copy_patient_and_study_information,
    _inverse_uid_lookup,
    _setup_file_meta,
)

if TYPE_CHECKING:
    from resmip.segmentation.segmentation import SegmentationCollection


def _get_series_frames(image_directory: Path):
    dicom_files = get_series_dicom_files(image_directory)
    if not dicom_files:
        raise ValueError(f"No valid image series files found in {image_directory}.")
    return [
        (ds.SOPInstanceUID, ds.ImagePositionPatient) for ds in (dcmread(f) for f in dicom_files)
    ]


def _create_code_sequence_item(code_value, scheme_designator, code_meaning):
    item = Dataset()
    item.CodeValue = code_value
    item.CodingSchemeDesignator = scheme_designator
    item.CodeMeaning = code_meaning
    return item


def _copy_series_data_from_reference(ds: Dataset, reference_ds: Dataset):
    """Copy SEG-specific tags that are not needed for RTSTRUCT.

    Args:
        ds (Dataset): Target SEG dataset to populate.
        reference_ds (Dataset): First slice of the reference series.
    """
    for keyword in ["SeriesDate", "ContentDate", "SeriesTime", "ContentTime"]:
        if keyword in reference_ds:
            setattr(ds, keyword, getattr(reference_ds, keyword))
    for keyword in ["FrameOfReferenceUID", "PositionReferenceIndicator"]:
        setattr(ds, keyword, getattr(reference_ds, keyword))


def _copy_patient_data_from_series(
    ds: Dataset, dicom_series_path: Path, segments_number: int
):  # pylint: disable=too-many-locals
    series_data = load_sorted_image_series(dicom_series_path)
    series_ds = series_data[0]

    _copy_series_data_from_reference(ds, series_ds)
    _copy_patient_and_study_information(ds, series_ds)

    series_slices = _get_series_frames(dicom_series_path)

    referenced_series_sop_class = series_ds.SOPClassUID
    # Referenced instances
    ref_instances = []
    for uid, _ in series_slices:
        inst = Dataset()
        inst.ReferencedSOPClassUID = referenced_series_sop_class
        inst.ReferencedSOPInstanceUID = uid
        ref_instances.append(inst)

    ref_series_item = Dataset()
    ref_series_item.ReferencedInstanceSequence = Sequence(ref_instances)
    ref_series_item.SeriesInstanceUID = series_ds.SeriesInstanceUID
    ds.ReferencedSeriesSequence = Sequence([ref_series_item])

    purpose_item = _create_code_sequence_item(
        "121322", "DCM", "Source image for image processing operation"
    )
    derivation_code_item = _create_code_sequence_item("113076", "DCM", "Segmentation")
    # Per-frame groups
    per_frame_seq = []
    for i, (src_uid, img_pos) in enumerate(series_slices, start=1):
        for seg_number in range(1, segments_number + 1):
            frame = Dataset()

            # Derivation Image
            src_img = Dataset()
            src_img.ReferencedSOPClassUID = referenced_series_sop_class
            src_img.ReferencedSOPInstanceUID = src_uid
            src_img.PurposeOfReferenceCodeSequence = Sequence([purpose_item])

            deriv_img = Dataset()
            deriv_img.SourceImageSequence = Sequence([src_img])
            deriv_img.DerivationCodeSequence = Sequence([derivation_code_item])
            frame.DerivationImageSequence = Sequence([deriv_img])

            # Frame Content
            frame_content = Dataset()
            frame_content.DimensionIndexValues = [seg_number, i]
            frame.FrameContentSequence = Sequence([frame_content])

            # Plane Position
            plane_pos = Dataset()
            plane_pos.ImagePositionPatient = img_pos
            frame.PlanePositionSequence = Sequence([plane_pos])

            # Segment Identification
            seg_id = Dataset()
            seg_id.ReferencedSegmentNumber = seg_number
            frame.SegmentIdentificationSequence = Sequence([seg_id])

            per_frame_seq.append(frame)
    ds.PerFrameFunctionalGroupsSequence = Sequence(per_frame_seq)


class SegDataset:
    """Abstraction of the ``pydicom.Dataset`` used for DICOM SEG."""

    def __init__(  # pylint: disable=too-many-statements
        self,
        segmentations: SegmentationCollection,
        dicom_series_path: Path,
        series_description: str = "",
    ):
        """Create dataset for DICOM SEG file."""
        image_size = segmentations.size
        image_spacing = segmentations.spacing
        image_direction = segmentations.direction

        ds = FileDataset(
            None,
            {},
            file_meta=_setup_file_meta(storage_type="Segmentation Storage"),
            preamble=b"\0" * 128,
        )

        ds.ImageType = ["DERIVED", "PRIMARY"]
        ds.SOPClassUID = _inverse_uid_lookup("Segmentation Storage")
        ds.SOPInstanceUID = generate_uid()
        _copy_patient_data_from_series(
            ds, dicom_series_path=dicom_series_path, segments_number=len(segmentations)
        )
        # ds.SeriesDate                  = image.metadata[string_tag_for_keyword("SeriesDate")]
        # ds.SeriesTime                  = image.metadata[string_tag_for_keyword("SeriesTime")]

        _set_content_datetime(ds)

        ds.AccessionNumber = ""
        ds.Modality = "SEG"
        ds.Manufacturer = "pymaitre"
        ds.ReferringPhysicianName = ""
        ds.SeriesDescription = series_description
        ds.ManufacturerModelName = "https://github.com/pymaitre/resmip.git"
        ds.DeviceSerialNumber = "0"
        ds.SoftwareVersions = __version__

        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = "1"
        ds.InstanceNumber = "1"

        dim_org_uid = generate_uid()

        dim_org_item = Dataset()
        dim_org_item.DimensionOrganizationUID = dim_org_uid
        ds.DimensionOrganizationSequence = Sequence([dim_org_item])

        # Dimension Index: ReferencedSegmentNumber
        dim_idx_seg = Dataset()
        dim_idx_seg.DimensionOrganizationUID = dim_org_uid
        dim_idx_seg.DimensionIndexPointer = Tag(0x0062, 0x000B)
        dim_idx_seg.FunctionalGroupPointer = Tag(0x0062, 0x000A)
        dim_idx_seg.DimensionDescriptionLabel = "ReferencedSegmentNumber"

        # Dimension Index: ImagePositionPatient
        dim_idx_pos = Dataset()
        dim_idx_pos.DimensionOrganizationUID = dim_org_uid
        dim_idx_pos.DimensionIndexPointer = Tag(0x0020, 0x0032)
        dim_idx_pos.FunctionalGroupPointer = Tag(0x0020, 0x9113)
        dim_idx_pos.DimensionDescriptionLabel = "ImagePositionPatient"

        ds.DimensionIndexSequence = Sequence([dim_idx_seg, dim_idx_pos])

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = image_size[1]
        ds.Columns = image_size[0]
        if segmentations.segmentation_type == SegmentationType.binary:
            ds.BitsAllocated = 1
            ds.BitsStored = 1
            ds.HighBit = 0
            ds.SegmentationType = "BINARY"
        else:
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.SegmentationType = "FRACTIONAL"
            ds.MaximumFractionalValue = 255
            ds.SegmentationFractionalType = "PROBABILITY"
        ds.PixelRepresentation = 0
        ds.LossyImageCompression = "00"

        ds.SegmentSequence = Sequence([])

        shared_fg = Dataset()

        plane_orient = Dataset()
        plane_orient.ImageOrientationPatient = list(image_direction[:6])
        shared_fg.PlaneOrientationSequence = Sequence([plane_orient])

        pixel_measures = Dataset()
        pixel_measures.SliceThickness = str(image_spacing[2])
        pixel_measures.SpacingBetweenSlices = str(image_spacing[2])
        pixel_measures.PixelSpacing = list(image_spacing[:2])
        shared_fg.PixelMeasuresSequence = Sequence([pixel_measures])

        ds.SharedFunctionalGroupsSequence = Sequence([shared_fg])

        ds.ContentLabel = "SEGMENTATION"
        ds.ContentDescription = ""

        self.dataset = ds

    def add_segment(self, name: str, number: int):
        """Add one segmentation to the dataset."""
        seg_item = Dataset()

        prop_cat_item = _create_code_sequence_item("85756007", "SCT", "Tissue")
        seg_item.SegmentedPropertyCategoryCodeSequence = Sequence([prop_cat_item])

        seg_item.SegmentNumber = number
        seg_item.SegmentLabel = name
        seg_item.SegmentAlgorithmType = "SEMIAUTOMATIC"
        seg_item.SegmentAlgorithmName = "SlicerEditor"
        seg_item.RecommendedDisplayCIELabValue = [41661, 41167, 40792]

        prop_type_item = _create_code_sequence_item("85756007", "SCT", name)
        seg_item.SegmentedPropertyTypeCodeSequence = Sequence([prop_type_item])

        self.dataset.SegmentSequence.append(seg_item)

    def save(self, file_path):
        """Save the DICOM dataset to file."""
        dcmwrite(file_path, self.dataset, enforce_file_format=True)
