"""DICOM SEG modality."""

from __future__ import annotations

import logging
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydicom
import SimpleITK as sitk

from resmip.segmentation.utils import SegmentationType
from resmip.utils import format_digit_string

from .annotation import (
    DicomAnnotation,
    _duplicate_annotation_names,
    _match_annotation_names,
)
from .seg_dataset import SegDataset

if TYPE_CHECKING:
    from resmip.segmentation.segmentation import SegmentationCollection

__all__ = []

logger = logging.getLogger(__name__)


def _compute_direction(dataset: pydicom.Dataset) -> list[int | float]:
    """Compute the 9-element image direction from a DICOM SEG dataset.

    Reads the row and column cosines from ``ImageOrientationPatient`` in the
    Shared Functional Groups Sequence and derives the normal vector as their
    cross product, returning a flattened row-major direction cosine matrix
    compatible with SimpleITK's ``SetDirection``.

    Args:
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file.

    Returns:
        list[int | float]: 9-element list [row_x, row_y, row_z,
            col_x, col_y, col_z, normal_x, normal_y, normal_z].
    """
    orientation = (
        dataset.SharedFunctionalGroupsSequence[0]
        .PlaneOrientationSequence[0]
        .ImageOrientationPatient
    )
    row_cosines = np.array(orientation[:3])
    col_cosines = np.array(orientation[3:])
    return [
        *row_cosines,
        *col_cosines,
        *np.cross(row_cosines, col_cosines),
    ]


def _compute_spacing(dataset: pydicom.Dataset, frames: list[int]) -> list[float]:
    """Compute voxel spacing from a DICOM SEG dataset for the given frames.

    In-plane (x, y) spacing is read from ``PixelSpacing`` in the Shared
    Functional Groups Sequence. Slice spacing is derived from the difference
    between consecutive ``ImagePositionPatient`` Z coordinates. If only a
    single frame is provided, ``SliceThickness`` is used as the Z spacing
    fallback.

    Args:
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file.
        frames (list[int]): Indices into ``PerFrameFunctionalGroupsSequence``
            corresponding to a single segment, in acquisition order.

    Raises:
        ValueError: If the Z spacing between consecutive frames is not uniform.

    Returns:
        list[float]: 3-element spacing list [col_spacing, row_spacing, z_spacing]
            in mm, compatible with SimpleITK's ``SetSpacing``.
    """
    positions = []
    for i in frames:
        pos = (
            dataset.PerFrameFunctionalGroupsSequence[i]
            .PlanePositionSequence[0]
            .ImagePositionPatient
        )
        positions.append(pos)
    positions = np.array(positions)
    if len(frames) == 1:
        z_spacing = float(
            dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        )
    else:
        offset = (positions[1:] - positions[:-1]).round(decimals=4)
        if np.unique(offset[:, :-1]) != 0:
            logger.warning("Some frames have a different x-y origin.")
        if len(np.unique(offset[:, -1])) > 1:
            logger.error("Nonuniform slice spacing.")
            raise ValueError
        z_spacing = np.unique(offset[:, -1])[0]
    xy_spacing = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    return [float(x) for x in (*xy_spacing[::-1], z_spacing)]


def _convert_single_segmentation(segment: pydicom.Dataset, dataset: pydicom.Dataset):
    """Read a single segmentation from a DICOM SEG dataset.

    Identifies the frames belonging to the given segment by matching
    ``ReferencedSegmentNumber`` in the Per-Frame Functional Groups Sequence,
    then reconstructs a SimpleITK image with the correct spatial metadata
    (origin, direction, spacing).

    Args:
        segment (pydicom.Dataset): Dataset item from ``SegmentSequence``
            describing the segment to extract, including its number and label.
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file,
            including pixel data and functional group sequences.

    Raises:
        KeyError: If no frames are found for the given segment number.

    Returns:
        DicomAnnotation: Object containing the segment label and its mask
            as a SimpleITK Image with spatial metadata set.
    """
    segment_frames = []
    for i, x in enumerate(dataset["PerFrameFunctionalGroupsSequence"]):
        if (
            x.SegmentIdentificationSequence[0].ReferencedSegmentNumber
            == segment["SegmentNumber"].value
        ):
            segment_frames.append(i)
    if len(segment_frames) == 0:
        logger.error("No frames found for segmentation %s.", segment.SegmentLabel)
        raise KeyError

    origin = (
        dataset.PerFrameFunctionalGroupsSequence[segment_frames[0]]
        .PlanePositionSequence[0]
        .ImagePositionPatient
    )
    direction = _compute_direction(dataset=dataset)
    spacing = _compute_spacing(dataset=dataset, frames=segment_frames)

    segmentation = sitk.GetImageFromArray(dataset.pixel_array[segment_frames])
    segmentation.SetOrigin(origin=origin)
    segmentation.SetDirection(direction=direction)
    segmentation.SetSpacing(spacing=spacing)
    return DicomAnnotation(name=segment.SegmentLabel, image=segmentation)


def _read(  # pylint: disable=too-many-locals
    seg_path: Path,
    structure_names: str | list[str] | None = None,
    parallel: bool = False,
    regex: bool = False,
    case_sensitive: bool = True,
) -> tuple[list[DicomAnnotation], dict[str, str]]:
    """Read segmentations from a DICOM SEG file.

    Reads all segments from the file by default. Optionally filters to a
    subset of segments by name, with support for exact matching, regular
    expressions, and case-insensitive comparison.

    Args:
        seg_path (Path): Path to the DICOM SEG file.
        structure_names (str | list[str] | None): Name(s) of the segments to
            read. If ``None``, all segments are returned. A single string is
            treated as a list of one name. Defaults to ``None``.
        parallel (bool): If ``True``, convert segments to SimpleITK images
            in parallel using a thread pool. Defaults to ``False``.
        regex (bool): If ``True``, treat ``structure_names`` as regular
            expressions. Defaults to ``False``.
        case_sensitive (bool): If ``False``, perform case-insensitive name
            matching. Defaults to ``True``.

    Raises:
        NotImplementedError: If the DICOM SEG file contains only a single frame.
        ValueError: If duplicate segment names are found in the file.

    Returns:
        tuple[list[DicomAnnotation], dict[str, str]]: A tuple of:
            - List of ``DicomAnnotation`` objects, one per matched segment.
            - Dictionary of DICOM metadata key-value pairs extracted from
              the first frame of the file.
    """
    dataset = pydicom.dcmread(seg_path, force=True)
    if dataset.pixel_array.ndim == 2:
        raise NotImplementedError("Single-frame DICOM SEG are not supported yet.")

    dicom_segments: pydicom.Sequence = dataset.SegmentSequence
    if _duplicate_annotation_names(
        dicom_segments, key="SegmentLabel", case_sensitive=case_sensitive
    ):
        raise ValueError("Multiple segments with the same name were found.")

    if isinstance(structure_names, str):
        structure_names = [structure_names]

    if structure_names is not None:
        matched_dicom_segments = _match_annotation_names(
            annotation_sequence=dicom_segments,
            names_to_match=structure_names,
            key="SegmentLabel",
            regex=regex,
            case_sensitive=case_sensitive,
        )
    else:
        matched_dicom_segments = dicom_segments
    if len(matched_dicom_segments) == 0:
        logger.warning("No matching segments found.")

    dicom_series_reader = sitk.ImageSeriesReader()
    dicom_series_reader.SetFileNames([seg_path])
    dicom_series_reader.MetaDataDictionaryArrayUpdateOn()
    dicom_series_reader.LoadPrivateTagsOn()
    _ = dicom_series_reader.Execute()
    seg_metadata = {}
    for key in dicom_series_reader.GetMetaDataKeys(0):
        value = dicom_series_reader.GetMetaData(0, key)
        value = format_digit_string(value)
        seg_metadata[key] = value

    segments = []
    if parallel is True:
        num_threads = max(len(matched_dicom_segments), 1)
        with ThreadPool(num_threads) as p:
            segments = p.map(
                partial(_convert_single_segmentation, dataset=dataset),
                matched_dicom_segments,
            )
    else:
        for matched_segment in matched_dicom_segments:
            segments.append(_convert_single_segmentation(dataset=dataset, segment=matched_segment))
    return segments, seg_metadata


def _write(
    segmentations: SegmentationCollection,
    save_path: Path,
    dcm_series_path: Path,
    series_description: str = "",
) -> None:
    """Write a collection of segmentations to a DICOM SEG file.

    Constructs a DICOM SEG dataset from the given segmentation collection,
    copying patient and study metadata from the reference DICOM series.
    Pixel data is interleaved across segments and slices in the order
    required by the DICOM SEG standard: for each slice, all segment frames
    appear consecutively before moving to the next slice.

    Binary segmentations are packed with ``numpy.packbits`` (little-endian
    bit order) before writing. Fractional segmentations are written as 8-bit
    unsigned integers.

    Args:
        segmentations (SegmentationCollection): Collection of segmentations to
            write. All segmentations must be spatially compatible (same size,
            spacing, direction, and origin).
        save_path (Path): Full path of the output DICOM SEG file.
        dcm_series_path (Path): Path of the directory containing the reference
            DICOM series used to populate patient, study, and frame-of-reference
            metadata.
        series_description (str): Value for the ``SeriesDescription`` tag in
            the output file. Defaults to an empty string.

    Raises:
        ValueError: If ``dcm_series_path`` is ``None``.
    """
    if dcm_series_path is None:
        raise ValueError("The path of the reference dicom series must be specified.")

    image_size = segmentations.size
    seg = SegDataset(
        segmentations, dicom_series_path=dcm_series_path, series_description=series_description
    )
    pixel_array_shape = (
        image_size[2] * len(segmentations),
        image_size[1],
        image_size[0],
    )
    pixel_data = np.zeros(shape=pixel_array_shape, dtype=np.uint8)
    segments_number = len(segmentations)
    for number, segmentation in enumerate(segmentations.values()):
        seg.add_segment(segmentation.name, number=number + 1)
        seg_array = segmentation.numpy()
        for slice_index in range(image_size[2]):
            pixel_data[number + slice_index * segments_number] = seg_array[slice_index]
    seg.dataset.NumberOfFrames = str(pixel_data.shape[0])
    if segmentations.segmentation_type == SegmentationType.binary:
        pixel_data = np.packbits(pixel_data, bitorder="little")
    seg.dataset.PixelData = pixel_data.tobytes()

    seg.save(save_path)
