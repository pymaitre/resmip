.. title::
   Dicom rtdose

Dicom RT Doses
==============

Import a Dicom RT Dose
----------------------

.. note::

    The DICOM images and doses used in this tutorial can be found in
    ``tests/Dicom/dicompyler_img`` (``ct.0.dcm`` and ``rtdose.dcm``).

.. note::

    There is currently no support for saving DICOM RT Dose files.

Dicom RT Dose files can be imported with `resmip` after having imported the referenced series:

.. testcode:: python
    :hide:

    from pathlib import Path

    from resmip.image import Image
    from resmip.dose import Dose


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "dicompyler_img"
    dicom_dose_path = Path.cwd().parent / "tests" / "Dicom" / "dicompyler_img" / "rtdose.dcm"

First of all, we define the path of the ditectory containing the Dicom CT and RT Structure Set:

.. code-block:: python

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_dose_path = "path/to/dicom/rtdose_file"

.. testcode:: python

    from resmip.image import Image
    from resmip.dose import Dose


    dicom_image = Image.read(dicom_ct_directory)
    dicom_dose = Dose.read(dicom_dose_path, reference_image = dicom_image)

The resulting object is a ``Dose`` object, with the same properties of an ``Image``.
The dose is automatically co-registered to the referenced series.

.. testcode:: python

    print(f"Dose origin: {dicom_dose.origin}, Series origin: {dicom_image.origin}")
    print(f"Dose spacing: {dicom_dose.spacing}, Series spacing: {dicom_image.spacing}")
    print(f"Dose orientation: {dicom_dose.direction}, Series orientation: {dicom_image.direction}")
    print(f"Dose size: {dicom_dose.size}, Series size: {dicom_image.size}")

.. testoutput:: python

    Dose origin: (-275.0, -524.0, 168.5593), Series origin: (-275.0, -524.0, 168.5593)
    Dose spacing: (1.074219, 1.074219, 1.0), Series spacing: (1.074219, 1.074219, 1.0)
    Dose orientation: (1.0, 0.0, 1.224647e-16, 0.0, 1.0, 0.0, -1.224647e-16, 0.0, 1.0), Series orientation: (1.0, 0.0, 1.224647e-16, 0.0, 1.0, 0.0, -1.224647e-16, 0.0, 1.0)
    Dose size: (512, 512, 1), Series size: (512, 512, 1)
