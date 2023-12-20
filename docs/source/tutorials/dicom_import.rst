Import a Dicom series
=====================

Any Dicom series (e.g.: CT) can be imported with srmip using the read_image method:

.. code-block:: python

    from srmip.image import Image

    dicom_ct_directory = "path/to/dicom/directory"

    dicom_image = Image().read_image()
