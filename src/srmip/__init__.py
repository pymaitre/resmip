"""srmip."""

import importlib.metadata as im

from .image import Image
from .rt_structure import RTStructure  # noqa: F401
from .rt_structure import RTStructureSet  # noqa: F401

__version__ = im.version(__package__)


def read_image(*args, **kwargs):
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
