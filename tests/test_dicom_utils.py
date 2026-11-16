"""Test module for DICOM utility functions."""

import numpy as np
import pytest

from resmip.dicom_utils.utils import _image_orientation_from_direction


def test_image_orientation_holds_row_and_column_cosines_from_direction_columns():
    """Image Orientation (Patient) is read from the columns of the direction matrix.

    The image axes are rotated by 90 degrees within the axial plane, so the direction matrix
    is not symmetric and its first two rows differ from its first two columns.
    """
    row_cosine = np.array([0.0, 1.0, 0.0])
    column_cosine = np.array([-1.0, 0.0, 0.0])
    slice_normal = np.cross(row_cosine, column_cosine)
    direction = np.column_stack([row_cosine, column_cosine, slice_normal]).flatten()
    image_orientation = _image_orientation_from_direction(tuple(direction))

    assert image_orientation == pytest.approx([*row_cosine, *column_cosine], abs=1e-6)
