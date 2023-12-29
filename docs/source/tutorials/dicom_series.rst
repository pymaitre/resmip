.. title::
   Dicom series

Import a Dicom series
=====================

Any Dicom series (e.g.: CT) can be imported with srmip using the ``read_image`` method from the ``Image`` class.

.. testcode:: python
    :hide:

    from pathlib import Path

    from srmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "siemens_mprage_0_dcm"

First of all, we define the path of the ditectory containing the Dicom CT:

.. code-block:: python

    dicom_ct_directory = "path/to/dicom/directory"

.. testcode:: python

    from srmip.image import Image


    dicom_image = Image().read_image(dicom_ct_directory)

Alternatively, images can be imported with the ``read_image`` method:

.. testcode:: python

    import srmip


    dicom_image = srmip.read_image(dicom_ct_directory)

Save a Dicom series
===================

Images can be saved to Dicom using the ``write_image`` method of ``Image``, as long as the destination path is a directory:

.. testcode:: python
    :hide:

    from pathlib import Path
    import tempfile

    from srmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "siemens_mprage_0_dcm"
    dicom_ct_destination_directory = tempfile.mkdtemp()
    #print(dicom_ct_destination_directory)

First of all, we define the path of the ditectory containing the Dicom CT and the directory where we want to save the Dicom files:

.. code-block:: python

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_ct_destination_directory = "path/to/new/dicom/directory"

.. testcode:: python

    from srmip.image import Image


    dicom_image = Image().read_image(dicom_ct_directory)
    dicom_image.write_image(dicom_ct_destination_directory)


Alternatively, images can be saved to dicom with the ``write_image`` method:

.. testcode:: python

    import srmip


    dicom_image = srmip.read_image(dicom_ct_directory)
    srmip.write_image(dicom_image, dicom_ct_destination_directory)


..
    Cleanup block below

.. testcode:: python
    :hide:

    import shutil

    shutil.rmtree(dicom_ct_destination_directory)
