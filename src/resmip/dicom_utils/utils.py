"""Utility functions for DICOM files."""

from collections.abc import Sequence
from datetime import datetime

from pydicom import Dataset

__all__ = []


def _image_orientation_from_direction(direction: Sequence[float]) -> list[float]:
    """Convert a SimpleITK direction matrix to an Image Orientation (Patient) value.

    SimpleITK stores the 3x3 direction matrix row-major, with the physical direction of image
    index axis j in column j. PS3.3 C.7.6.2.1.1 defines Image Orientation (Patient) as the row
    cosine (the direction of increasing column index, image axis x) followed by the column
    cosine (the direction of increasing row index, image axis y). The value is therefore
    columns 0 and 1 of the matrix, not its first six elements: those are rows 0 and 1, which
    coincide with the columns only when the matrix is symmetric, as for the identity.

    Args:
        direction (Sequence[float]): Flattened 9-element row-major direction matrix, as
            returned by ``sitk.Image.GetDirection()``.

    Returns:
        list[float]: The six direction cosines, row cosine first, then column cosine.
    """
    return [*direction[0::3], *direction[1::3]]


def _set_content_datetime(ds: Dataset, dt: datetime | None = None):
    """Set ContentDate and ContentTime tags on a DICOM dataset.

    Args:
        ds (Dataset): Target dataset to populate.
        dt (datetime | None): Datetime to use. If ``None``, uses the current
            local time.
    """
    if dt is None:
        dt = datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")
