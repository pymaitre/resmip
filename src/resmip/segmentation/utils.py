"""Utility functions and classes for Segmentation objects."""

from enum import Enum

__all__ = ["SegmentationType"]


class SegmentationType(Enum):
    """Type of encoding used for the segmented property.

    The type of encoding used to indicate the presence of the
    segmented property at a pixel/voxel location.
    """

    binary = "BINARY"
    """Each voxel is either part of the segment or not."""
    fractional = "FRACTIONAL"
    """Each voxel value represents a probability or occupancy percentage."""
