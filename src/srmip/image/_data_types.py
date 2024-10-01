"""Data types used for images."""

from typing import Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

ImageDTypeLike = Union[str, int, npt.DTypeLike]
"""Data types used for images."""


def _sitk_image_dtype(dtype: ImageDTypeLike):
    """
    Convert dtype to one of the supported sitk values.

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
