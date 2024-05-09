"""srmip."""

import importlib.metadata as im

from srmip.image import Image

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
