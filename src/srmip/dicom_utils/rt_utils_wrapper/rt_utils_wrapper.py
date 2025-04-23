"""Wrapper module for rt_utils, used when converting nifti files to DICOM."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pydicom
import skimage.measure


def create_contour(series_slice: pydicom.Dataset, contour_data: np.ndarray) -> pydicom.Dataset:
    """Create contours for a specific DICOM series slice."""
    contour_image = pydicom.Dataset()
    contour_image.ReferencedSOPClassUID = series_slice.SOPClassUID
    contour_image.ReferencedSOPInstanceUID = series_slice.SOPInstanceUID

    # Contour Image Sequence
    contour_image_sequence = pydicom.Sequence()
    contour_image_sequence.append(contour_image)

    contour = pydicom.Dataset()
    contour.ContourImageSequence = contour_image_sequence
    contour.ContourGeometricType = "CLOSED_PLANAR"  # TODO figure out how to get this value
    contour.NumberOfContourPoints = len(contour_data) // 3  # Each point has an x, y, and z value
    contour.ContourData = contour_data.tolist()

    return contour


def get_polygon_contours_from_slice_mask(slice_mask: np.ndarray) -> Tuple[np.ndarray]:
    """
    Convert the slice mask to a collection of polygon contours.

    :param slice_mask: Mask of the slice to be converted, of shape (x_dim, y_dim).
    :type slice_mask: np.ndarray
    :return: Tuple of polygon vertices (as x, y tuples) of the mask contour, of shape (n_points, 2).
        The length of the tuple is the number of polygons in the slice.
    :rtype: Tuple[np.ndarray]
    """
    polygons = tuple(
        np.flip(np.array(poly), axis=1) - 1
        for poly in skimage.measure.find_contours(np.pad(slice_mask, 1).astype(np.uint8))
    )
    # Add the first point after the last one, in order to fully
    # close the polygon, otherwise, in case of contours with holes,
    # a small slice on the xy plane would be skipped
    return tuple(np.concatenate((poly, [poly[0]])) for poly in polygons)


def get_contour_from_slice_mask(slice_mask: np.ndarray) -> np.ndarray:
    """
    Convert the slice mask to a polygon contour.

    :param slice_mask: Mask of the slice to be converted, of shape (x_dim, y_dim).
    :type slice_mask: np.ndarray
    :return: Polygon vertices (as x, y tuples) of the mask contour, of shape (n_points, 2).
    :rtype: np.ndarray
    """
    polygons = get_polygon_contours_from_slice_mask(slice_mask)
    # Connect each polygon with the first point of the first polygon,
    # in order to fully separate them
    corrected_polygons = tuple(
        np.concatenate((poly, [polygons[0][0]]), axis=0) for poly in polygons
    )
    return np.concatenate(corrected_polygons, axis=0)
