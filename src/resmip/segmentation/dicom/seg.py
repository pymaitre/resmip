"""DICOM SEG modality."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


_FRAME_GRID_RESIDUAL_TOLERANCE = 1e-3
"""Largest distance, in voxels, between a frame position and the grid shared by all frames.

Frame positions pass through 16-character Decimal String elements, so a frame on the grid in
intent is off by floating-point noise in practice. A larger distance means the frames do not
lie on one regularly spaced grid, which a single SimpleITK image cannot represent.
"""

_SLICE_SPACING_DECIMALS = 4
"""Decimal places, in mm, kept in a slice spacing derived from frame positions.

A spacing derived from frame positions is a difference of Decimal String values and carries
floating-point noise. Rounding to 0.1 µm restores the intended value, as the previous reader
did, while staying far below any real slice spacing.
"""


@dataclass(frozen=True)
class _FrameGrid:
    """Voxel grid shared by every segment of a DICOM SEG, and the slice holding each frame.

    One grid for the whole file keeps all segments spatially compatible even when each
    segment stores frames on a different subset of slices, as writers that omit empty frames
    produce. Slices that no segment stores are part of the grid and read back as zeros.
    """

    origin: tuple[float, float, float]
    """Physical position of the first voxel, in mm."""
    spacing: tuple[float, float, float]
    """Voxel spacing in mm, in (x, y, z) order."""
    direction: tuple[float, ...]
    """Grid direction.

    Flattened row-major direction matrix whose columns are the row cosine,
    the column cosine and the slice normal, as ``sitk.Image.SetDirection`` expects.
    """
    slice_count: int
    """Number of slices spanned by all frames."""
    frame_slice_indices: np.ndarray
    """Slice index of each frame, in frame order."""


def _compute_slice_spacing(dataset: pydicom.Dataset, slice_coordinates: np.ndarray) -> float:
    """Compute the distance between adjacent slices of a DICOM SEG.

    ``SpacingBetweenSlices`` is used when present. Otherwise the spacing is the smallest gap
    between distinct frame positions along the slice normal, which is the true spacing as
    long as at least two adjacent slices are stored. With a single distinct position there is
    no gap to measure, and ``SliceThickness`` is the only available estimate.

    Args:
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file.
        slice_coordinates (np.ndarray): Position of each frame along the slice normal, in mm.

    Raises:
        ValueError: If the resulting spacing is not positive.

    Returns:
        float: Slice spacing in mm.
    """
    pixel_measures = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
    if "SpacingBetweenSlices" in pixel_measures:
        slice_spacing = float(pixel_measures.SpacingBetweenSlices)
    else:
        distinct_coordinates = np.unique(np.round(slice_coordinates, _SLICE_SPACING_DECIMALS))
        if len(distinct_coordinates) == 1:
            slice_spacing = float(pixel_measures.SliceThickness)
        else:
            slice_spacing = round(
                float(np.diff(distinct_coordinates).min()), _SLICE_SPACING_DECIMALS
            )
    if slice_spacing <= 0:
        raise ValueError(f"DICOM SEG slice spacing must be positive, found {slice_spacing} mm.")
    return slice_spacing


def _compute_frame_grid(dataset: pydicom.Dataset) -> _FrameGrid:
    """Compute the voxel grid shared by all frames of a DICOM SEG.

    Each frame is placed by projecting its ``ImagePositionPatient`` on the slice normal, the
    cross product of the row and column cosines of ``ImageOrientationPatient``. Projecting on
    the normal rather than reading patient z keeps sagittal, coronal and oblique slices
    correct. The grid starts at the lowest position along the normal, so slices are ordered
    along the normal whatever the frame order in the file.

    Args:
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file.

    Returns:
        _FrameGrid: Grid shared by all segments, with the slice index of each frame.
    """
    shared_groups = dataset.SharedFunctionalGroupsSequence[0]
    orientation = shared_groups.PlaneOrientationSequence[0].ImageOrientationPatient
    row_cosine = np.array(orientation[:3], dtype=float)
    column_cosine = np.array(orientation[3:], dtype=float)
    slice_normal = np.cross(row_cosine, column_cosine)
    direction_matrix = np.column_stack([row_cosine, column_cosine, slice_normal])

    frame_positions = np.array(
        [
            frame_groups.PlanePositionSequence[0].ImagePositionPatient
            for frame_groups in dataset.PerFrameFunctionalGroupsSequence
        ],
        dtype=float,
    )
    slice_coordinates = frame_positions @ slice_normal
    slice_spacing = _compute_slice_spacing(dataset, slice_coordinates)
    frame_slice_indices = np.rint(
        (slice_coordinates - slice_coordinates.min()) / slice_spacing
    ).astype(int)
    origin = frame_positions[np.argmin(slice_coordinates)]

    row_spacing, column_spacing = (
        float(value) for value in shared_groups.PixelMeasuresSequence[0].PixelSpacing
    )
    spacing_xyz = (column_spacing, row_spacing, slice_spacing)

    grid_positions = origin + np.outer(frame_slice_indices * slice_spacing, slice_normal)
    voxel_residuals = (frame_positions - grid_positions) @ direction_matrix / np.array(spacing_xyz)
    largest_residual = float(np.abs(voxel_residuals).max())
    if largest_residual > _FRAME_GRID_RESIDUAL_TOLERANCE:
        raise ValueError(
            "DICOM SEG frames do not lie on one regular grid: a frame is "
            f"{largest_residual:.3g} voxels away from it, above the tolerance of "
            f"{_FRAME_GRID_RESIDUAL_TOLERANCE}."
        )

    return _FrameGrid(
        origin=tuple(float(value) for value in origin),
        spacing=spacing_xyz,
        direction=tuple(float(value) for value in direction_matrix.flatten()),
        slice_count=int(frame_slice_indices.max()) + 1,
        frame_slice_indices=frame_slice_indices,
    )


def _convert_single_segmentation(
    segment: pydicom.Dataset,
    dataset: pydicom.Dataset,
    frame_grid: _FrameGrid,
    pixel_array: np.ndarray,
) -> DicomAnnotation:
    """Read a single segmentation from a DICOM SEG dataset.

    Identifies the frames belonging to the given segment by matching
    ``ReferencedSegmentNumber`` in the Per-Frame Functional Groups Sequence and places each
    one on its slice of the shared grid. Slices with no frame for this segment stay zero.

    Args:
        segment (pydicom.Dataset): Dataset item from ``SegmentSequence``
            describing the segment to extract, including its number and label.
        dataset (pydicom.Dataset): Dataset containing the whole DICOM SEG file.
        frame_grid (_FrameGrid): Grid shared by all segments of the file.
        pixel_array (np.ndarray): Decoded frames of the file, in (frame, row, column) order.

    Raises:
        KeyError: If no frames are found for the given segment number.
        ValueError: If two frames of the segment lie on the same slice.

    Returns:
        DicomAnnotation: Object containing the segment label and its mask
            as a SimpleITK Image with spatial metadata set.
    """
    segment_frames = [
        frame_index
        for frame_index, frame_groups in enumerate(dataset.PerFrameFunctionalGroupsSequence)
        if frame_groups.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        == segment.SegmentNumber
    ]
    if len(segment_frames) == 0:
        logger.error("No frames found for segmentation %s.", segment.SegmentLabel)
        raise KeyError
    segment_slice_indices = frame_grid.frame_slice_indices[segment_frames]
    if len(np.unique(segment_slice_indices)) != len(segment_slice_indices):
        raise ValueError(
            f"Segment {segment.SegmentLabel} has more than one frame on the same slice."
        )

    segment_array = np.zeros(
        shape=(frame_grid.slice_count, *pixel_array.shape[1:]), dtype=pixel_array.dtype
    )
    segment_array[segment_slice_indices] = pixel_array[segment_frames]

    segmentation = sitk.GetImageFromArray(segment_array)
    segmentation.SetOrigin(frame_grid.origin)
    segmentation.SetDirection(frame_grid.direction)
    segmentation.SetSpacing(frame_grid.spacing)
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
    pixel_array = dataset.pixel_array
    if pixel_array.ndim == 2:
        raise NotImplementedError("Single-frame DICOM SEG are not supported yet.")
    frame_grid = _compute_frame_grid(dataset)

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
                partial(
                    _convert_single_segmentation,
                    dataset=dataset,
                    frame_grid=frame_grid,
                    pixel_array=pixel_array,
                ),
                matched_dicom_segments,
            )
    else:
        for matched_segment in matched_dicom_segments:
            segments.append(
                _convert_single_segmentation(
                    segment=matched_segment,
                    dataset=dataset,
                    frame_grid=frame_grid,
                    pixel_array=pixel_array,
                )
            )
    return segments, seg_metadata


def _select_frames_to_keep(frames: np.ndarray, segment_count: int) -> np.ndarray:
    """Select the frames of a DICOM SEG that hold at least one segmented pixel.

    Frames are in the writer's order: slice-major, segment-minor, so frame
    ``slice_index * segment_count + segment_index`` belongs to segment ``segment_index``.

    A segment with no segmented pixel on any slice keeps its all-zero frame on slice 0. The
    reader finds a segment's frames through ``ReferencedSegmentNumber`` and raises
    ``KeyError`` for a segment with none, and whether a segment with zero frames conforms has
    not been checked with ``dciodvfy``. One empty frame keeps the segment readable and the
    file valid.

    Args:
        frames (np.ndarray): Frames in writer order, in (frame, row, column) order.
        segment_count (int): Number of segments interleaved in ``frames``.

    Returns:
        np.ndarray: Boolean mask over the frames, ``True`` for each frame to write.
    """
    kept_by_slice_and_segment = (
        frames.reshape(len(frames), -1).any(axis=1).reshape(-1, segment_count)
    )
    segments_without_pixels = ~kept_by_slice_and_segment.any(axis=0)
    kept_by_slice_and_segment[0, segments_without_pixels] = True
    return kept_by_slice_and_segment.reshape(-1)


def _write(
    segmentations: SegmentationCollection,
    save_path: Path,
    dcm_series_path: Path,
    series_description: str = "",
    *,
    omit_empty_frames: bool = False,
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
        omit_empty_frames (bool): If ``True``, write only the frames that hold at least one
            segmented pixel. A segment with none keeps one empty frame on the first slice.
            Defaults to ``False``.

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
    if omit_empty_frames:
        kept_frames = _select_frames_to_keep(pixel_data, segments_number)
        pixel_data = pixel_data[kept_frames]
        seg.dataset.PerFrameFunctionalGroupsSequence = pydicom.Sequence(
            [
                frame_groups
                for frame_groups, keep in zip(
                    seg.dataset.PerFrameFunctionalGroupsSequence, kept_frames, strict=True
                )
                if keep
            ]
        )
    seg.dataset.NumberOfFrames = str(pixel_data.shape[0])
    if segmentations.segmentation_type == SegmentationType.binary:
        pixel_data = np.packbits(pixel_data, bitorder="little")
    seg.dataset.PixelData = pixel_data.tobytes()

    seg.save(save_path)
