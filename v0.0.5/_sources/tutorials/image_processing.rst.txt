.. title::
   Image processing

Image processing
================

.. note::

    The DICOM images used in this tutorial can be found in ``tests/Dicom/IBSI1_CT_phantom/CT_00000``.

.. testcode:: python
    :hide:

    from pathlib import Path

    from resmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "IBSI1_CT_phantom" / "CT_00000"
    image = Image.read_image(dicom_ct_directory)

Voxel spacing
-------------

After reading an image, its voxel spacing in mm is stored in the ``spacing`` property:

.. testcode:: python

    print(image.spacing)

.. testoutput:: python

    (0.97699999809265, 0.97699999809265, 3.0)

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

    Original image size: (204, 201, 60)
    Resampled image size: (399, 393, 360)

Image origin
------------

The origin of the image (in mm) can be obtained from the ``origin``
property as a (x,y,z) tuple.

.. testcode:: python

    print(image.origin)

.. testoutput:: python

    (-174.39450073242, -79.625503540039, -100.40000152587)

The origin can be changed with the following code:

.. testcode:: python

    from copy import copy

    shifted_image = copy(image)
    shifted_image.origin = (1, 1, 1)
    print(shifted_image.origin)

.. testoutput:: python

    (1, 1, 1)

.. warning::

    Changing the origin of the image causes a shift of the pixel data,
    effectively translating the whole image. This can cause unexpected
    behaviours when overlapping the image with other images (e.g.: segmentations)
    and when they are saved to file.

    If the intent of changing the origin is to translate the image, use
    ``Image.pad(...)`` instead, if applicable.

Image orientation
-----------------

Image orientation can be accessed from the ``direction`` property:

.. testcode:: python

    print(image.direction)

.. testoutput:: python

    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

The resulting tuple is the direction cosine matrix.
For more information refer to SimpleITK documentation:
https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html#images
