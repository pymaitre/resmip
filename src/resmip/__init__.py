"""resmip."""

from ._version import __version__
from .dicom_utils import (
    get_contour_from_slice_mask,
    get_polygon_contours_from_slice_mask,
)
from .dose import Dose
from .image import CoregistrationMetric, DicomModality, Image, ImageDTypeLike
from .rt_structure import RTStructure
from .segmentation import (
    RTStructureSet,
    Segmentation,
    SegmentationCollection,
    SegmentationType,
)

__all__ = [
    "CoregistrationMetric",
    "DicomModality",
    "Dose",
    "get_contour_from_slice_mask",
    "get_polygon_contours_from_slice_mask",
    "Image",
    "ImageDTypeLike",
    "read_image",
    "read_structure",
    "read_structure_set",
    "RTStructure",
    "RTStructureSet",
    "Segmentation",
    "SegmentationCollection",
    "SegmentationType",
    "write_image",
    "__version__",
]


def read_image(*args, **kwargs) -> Image:
    """Call ``Image.read`` directly from a function.

    Done in the most similar way we do for ``SimpleITK.read_image``.
    """
    return Image.read(*args, **kwargs)


def write_image(image: Image, *args, **kwargs):
    """Call ``image.write`` directly from a function.

    Done in the most similar way we do for ``SimpleITK.write_image``.
    """
    return image.write(*args, **kwargs)


def read_structure(*args, **kwargs) -> RTStructure:
    """Call ``RTStructure.read`` directly from a function."""
    return RTStructure.read(*args, **kwargs)


def read_structure_set(*args, **kwargs) -> RTStructureSet:
    """Call ``RTStructureSet.read`` directly from a function."""
    return RTStructureSet.read(*args, **kwargs)
