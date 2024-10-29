.. title::
   Image processing

Image processing
=======================

.. note::

    The DICOM images used in this tutorial can be found in ``tests/Dicom/IBSI1_CT_phantom/CT_00000``.

.. testcode:: python
    :hide:

    from pathlib import Path

    from srmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "siemens_mprage_0_dcm"
    image = Image.read_image(dicom_ct_directory)

Change voxel spacing
--------------------

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
