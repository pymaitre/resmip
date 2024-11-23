.. title::
   Image processing

Image processing
================

.. note::

    The DICOM images used in this tutorial can be found in ``tests/Dicom/IBSI1_CT_phantom/CT_00000``.

.. testcode:: python
    :hide:

    from pathlib import Path

    from srmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "siemens_mprage_0_dcm"
    image = Image.read_image(dicom_ct_directory)

Voxel spacing
-------------

After reading an image, its voxel spacing in mm is stored in the ``spacing`` property:

.. testcode:: python

    print(image.spacing)

.. testoutput:: python

    (1.0, 1.0, 0.9241545893719807)

The image can be resampled to a different spacing with the following code:

.. testcode:: python

    new_spacing = (0.5, 0.5, 0.5)
    resampled_image = image.resample(new_spacing)
    print(resampled_image.spacing)

.. testoutput:: python

    (0.5, 0.5, 0.5)

As a consequence, the size of the image changes accordingly:

.. testcode:: python

    print(f"Original image size: {image.GetSize()}")
    print(f"Resampled image size: {resampled_image.GetSize()}")

.. testoutput:: python

    Original image size: (224, 224, 208)
    Resampled image size: (448, 448, 385)

Image origin
------------

The origin of the image (in mm) can be obtained from the ``origin``
property as a (x,y,z) tuple.

.. testcode:: python

    print(image.origin)

.. testoutput:: python

    (-106.14937655628, -165.42824882477, -65.187102036914)

The origin can be changed with the following code:

.. testcode:: python

    from copy import copy

    shifted_image = copy(image)
    shifted_image.origin = (1, 1, 1)
    print(shifted_image.origin)

.. testoutput:: python

    (1, 1, 1)

.. warning::

    Changing the origin of the image, causes a shift of the pixel data,
    effectively translating the whole image. This can cause unexpected
    behaviours when overlapping the image with other images (e.g.: segmentations)
    and when they are saved to file.

    If the intent of changing the origin is to translate the image, use
    ``Image.pad(...)`` instead, if applicable.
