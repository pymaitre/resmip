"""Data types used for images."""

from typing import Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

__all__ = ["ImageDTypeLike"]

ImageDTypeLike = Union[str, int, npt.DTypeLike]
"""Data types used for images."""


def _sitk_image_dtype(dtype: ImageDTypeLike) -> int:
    """Convert dtype to one of the supported sitk values.

    If a sitk type is provided, do nothing.
    """
    if isinstance(dtype, int):
        return dtype

    conversion_map = {
        np.int8: sitk.sitkInt8,
        np.uint8: sitk.sitkUInt8,
        np.int16: sitk.sitkInt16,
        np.uint16: sitk.sitkUInt16,
        np.int32: sitk.sitkInt32,
        np.uint32: sitk.sitkUInt32,
        np.int64: sitk.sitkInt64,
        np.uint64: sitk.sitkUInt64,
        np.float32: sitk.sitkFloat32,
        np.float64: sitk.sitkFloat64,
        np.complex64: sitk.sitkComplexFloat32,
        np.complex128: sitk.sitkComplexFloat64,
    }

    numpy_dtype = getattr(np, np.dtype(dtype).name)
    try:
        return conversion_map[numpy_dtype]
    except KeyError as e:
        raise ValueError(
            f"The provided data type ({dtype}) is not supported as a SimpleITK type."
        ) from e


def _is_unsigned(dtype: ImageDTypeLike) -> bool:
    """Whether the type is unsigned or not."""
    usigned_types_map = {
        sitk.sitkInt8: False,
        sitk.sitkUInt8: True,
        sitk.sitkInt16: False,
        sitk.sitkUInt16: True,
        sitk.sitkInt32: False,
        sitk.sitkUInt32: True,
        sitk.sitkInt64: False,
        sitk.sitkUInt64: True,
        sitk.sitkFloat32: False,
        sitk.sitkFloat64: False,
        sitk.sitkComplexFloat32: False,
        sitk.sitkComplexFloat64: False,
    }
    return usigned_types_map[_sitk_image_dtype(dtype)]
