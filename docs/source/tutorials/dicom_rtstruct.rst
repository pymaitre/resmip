.. title::
   Dicom rtstruct

Dicom RT Structure Sets
=======================

Import a Dicom RT Structure Set
-------------------------------

.. note::

    The DICOM images and structures used in this tutorial can be found in
    ``tests/Dicom/IBSI1_CT_phantom`` (``CT_00000`` and ``RTst_00000/DCM_RS_00060.dcm``).

Dicom RT Structure Sets can be imported with `srmip` after having imported the referenced series:

.. testcode:: python
    :hide:

    from pathlib import Path

    from srmip.image import Image
    from srmip.rt_structure import RTStructureSet


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "IBSI1_CT_phantom" / "CT_00000"
    dicom_rtst_path = Path.cwd().parent / "tests" / "Dicom" / "IBSI1_CT_phantom" / "RTst_00000" / "DCM_RS_00060.dcm"

First of all, we define the path of the ditectory containing the Dicom CT and RT Structure Set:

.. code-block:: python

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_rtst_path = "path/to/dicom/rtstruct_file"

.. testcode:: python

    from srmip.image import Image
    from srmip.rt_structure import RTStructureSet


    dicom_image = Image().read_image(dicom_ct_directory)
    dicom_rtst = RTStructureSet().read_image(dicom_rtst_path, reference_image = dicom_image)

The resulting object is a dictionary containing all RT Structures found, indexed by their name.

.. testcode:: python

    print(dicom_rtst.keys())

.. testoutput:: python

    dict_keys(['GTV-1'])

Read only desired Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to speed-up the conversion of dicom RT Structures to nifti, a structure name (or list of structure names)
can be passed to ``read_image()``. The conversion is single-threaded by default, but it can be done in parallel.

.. testcode:: python

    dicom_rtst = RTStructureSet().read_image(dicom_rtst_path, structure_names = "GTV-1", reference_image = dicom_image)
    print(dicom_rtst.keys())

.. testoutput:: python

    dict_keys(['GTV-1'])

Regular expressions are supported too.

.. testcode:: python

    dicom_rtst = RTStructureSet().read_image(dicom_rtst_path, structure_names = "GTV-\d+", regex = True, reference_image = dicom_image)
    print(dicom_rtst.keys())

.. testoutput:: python

    dict_keys(['GTV-1'])

Save a Dicom RT Structure Set
-----------------------------

RT Structure Sets can be saved to Dicom using the ``write_image`` method of ``RTStructureSet``.
First of all, a reference dicom image is required. When dealing with nifti images, the image needs to be converted to dicom first.

.. testcode:: python
    :hide:

    from pathlib import Path
    import tempfile

    from srmip.image import Image


    dicom_ct_directory = Path.cwd().parent / "tests" / "Dicom" / "IBSI1_CT_phantom" / "CT_00000"
    dicom_rtst_path = Path.cwd().parent / "tests" / "Dicom" / "IBSI1_CT_phantom" / "RTst_00000" / "DCM_RS_00060.dcm"
    dicom_ct_destination_directory = tempfile.mkdtemp()
    dicom_rtst_destination_file = Path(dicom_ct_destination_directory) / "rtst.dcm"

.. code-block:: python

    dicom_ct_directory = "path/to/dicom/directory"
    dicom_rtst_path = "path/to/dicom/rtstruct_file"
    dicom_ct_destination_directory = "path/to/new/dicom/directory"
    dicom_rtst_destination_file = "path/to/new/dicom/rtstruct_file.dcm"

.. testcode:: python

    from srmip.image import Image
    from srmip.rt_structure import RTStructureSet


    dicom_image = Image().read_image(dicom_ct_directory)
    dicom_rtst = RTStructureSet().read_image(dicom_rtst_path, reference_image = dicom_image)

    # Save image to dicom
    dicom_image.write_image(dicom_ct_destination_directory)
    # Save RT Structure Set
    dicom_rtst.write_image(dicom_rtst_destination_file, reference_image_path = dicom_ct_destination_directory)

Customize DICOM output
^^^^^^^^^^^^^^^^^^^^^^

Set ``SeriesDescription``
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SeriesDescription`` for the saved DICOM RT Structure Set can be customized with
the ``series_description`` keyword:

.. testcode:: python

    # Save RT Structure Set with SeriesDescription
    dicom_rtst.write_image(dicom_rtst_destination_file, reference_image_path = dicom_ct_destination_directory, series_description = "CustomSeriesDescription")

..
    Cleanup block below

.. testcode:: python
    :hide:

    import shutil

    shutil.rmtree(dicom_ct_destination_directory)
