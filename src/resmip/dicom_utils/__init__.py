"""Conversion between DICOM and other ITK formats (e.g.: nifti)."""

from .constants import string_tag_for_keyword
from .rt_utils_wrapper import (
    get_contour_from_slice_mask,
    get_polygon_contours_from_slice_mask,
)

__all__ = [
    "get_contour_from_slice_mask",
    "get_polygon_contours_from_slice_mask",
    "string_tag_for_keyword",
]
