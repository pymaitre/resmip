.. title::
   Dicom series

Import a Dicom series
=====================

Any Dicom series (e.g.: CT) can be imported with srmip using the ``read_image`` method from the ``Image`` class:

.. code-block:: python

    from srmip.image import Image

    dicom_ct_directory = "path/to/dicom/directory"

    dicom_image = Image().read_image(dicom_ct_directory)

Alternatively, images can be imported with the ``read_image`` method:

.. code-block:: python

    import srmip

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_image = srmip.read_image(dicom_ct_directory)

Save a Dicom series
===================

Images can be saved to Dicom using the ``write_image`` method of ``Image``, as long as the destination path is a directory:

.. code-block:: python

    from srmip.image import Image

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_ct_destination_directory = "path/to/new/dicom/directory"

    dicom_image = Image().read_image(dicom_ct_directory)
    dicom_image.write_image(dicom_ct_destination_directory)


Alternatively, images can be saved to dicom with the ``write_image`` method:

.. code-block:: python

    import srmip

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_ct_destination_directory = "path/to/new/dicom/directory"

    dicom_image = srmip.read_image(dicom_ct_directory)
    srmip.write_image(dicom_image, dicom_ct_destination_directory)
