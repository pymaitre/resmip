"""Utility functions for DICOM files."""

from datetime import datetime

from pydicom import Dataset

__all__ = []


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
