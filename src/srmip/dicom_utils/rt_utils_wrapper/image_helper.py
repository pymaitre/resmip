"""Wrapper module from rt-utils."""

import os
from typing import List

import numpy as np
from pydicom import Dataset, dcmread
from pydicom.errors import InvalidDicomError


def load_sorted_image_series(dicom_series_path: str):
    """File contains helper methods for loading/formatting DICOM images and contours."""
    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise FileNotFoundError("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=get_slice_position, reverse=False)

    return series_data


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    """Load all DICOM images from the specified path."""
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, "pixel_array"):
                    series_data.append(ds)
            except InvalidDicomError:
                continue
    return series_data


def get_slice_position(series_slice: Dataset):
    """Get (x,y,)z position of the slice."""
    _, _, slice_direction = get_slice_directions(series_slice)
    return np.dot(slice_direction, series_slice.ImagePositionPatient)


def get_slice_directions(series_slice: Dataset):
    """Get direction of the slice."""
    orientation = series_slice.ImageOrientationPatient
    row_direction = np.array(orientation[:3])
    column_direction = np.array(orientation[3:])
    slice_direction = np.cross(row_direction, column_direction)

    if not np.allclose(np.dot(row_direction, column_direction), 0.0, atol=1e-3) or not np.allclose(
        np.linalg.norm(slice_direction), 1.0, atol=1e-3
    ):
        raise ValueError("Invalid Image Orientation (Patient) attribute")

    return row_direction, column_direction, slice_direction
