"""srmip."""

import importlib.metadata as im

from .dicom_nifti_conversion import DICOM_FIELDS  # noqa: F401
from .image import Image
from .rt_structure import RTStructure, RTStructureSet

__version__ = im.version(__package__)


def read_image(*args, **kwargs) -> Image:
    """
    Call Image().read_image directly from a function.

    Done in the most similar way we do for SimpleITK.read_image.
    """
    return Image().read_image(*args, **kwargs)


def write_image(image: Image, *args, **kwargs):
    """
    Call image.write_image directly from a function.

    Done in the most similar way we do for SimpleITK.write_image.
    """
    return image.write_image(*args, **kwargs)


def read_structure(*args, **kwargs) -> RTStructure:
    """Call RTStructure().read_image directly from a function."""
    return RTStructure().read_image(*args, **kwargs)


def read_structure_set(*args, **kwargs) -> RTStructureSet:
    """Call Image().read_image directly from a function."""
    return RTStructureSet().read_image(*args, **kwargs)
