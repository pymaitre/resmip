"""Test module for rt_structure.py"""

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

import srmip
from srmip.rt_structure.rt_structure import RTStructure, RTStructureSet

from .utils import (
    dicom_ct_path,
    dicom_rtst_path,
    dicom_rtst_path_with_hole,
    ibsi_rtst_path,
)

TEST_RTST_SIZE = (204, 201, 60)
"""Size of the test RT Structure."""
TEST_RTST_SPACING = (0.97699999809265, 0.97699999809265, 2.9999999999998486)
"""Spacing of the test RT Structure."""


def test_read_single_dicom_structure():
    """Read a dicom RT Structure from file."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )
    assert rtst.name == structure_name
    # check size
    assert rtst.GetSize() == TEST_RTST_SIZE
    # check spacing
    np.testing.assert_allclose(rtst.GetSpacing(), TEST_RTST_SPACING)


def test_read_single_dicom_structure_function():
    """Read a dicom RT Structure from file using the function in __init__.py."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = srmip.read_structure(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )
    reference_rtst = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )
    assert rtst.name == reference_rtst.name
    assert rtst.GetSize() == reference_rtst.GetSize()
    assert rtst.spacing == reference_rtst.spacing
    assert rtst.origin == reference_rtst.origin
    assert rtst.direction == reference_rtst.direction
    assert np.all(rtst.numpy() == reference_rtst.numpy())


def test_read_single_dicom_structure_wrong_name():
    """Read a dicom RT Structure from file, with a wrong name."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "gtv-1"
    with pytest.raises(IndexError):
        _ = RTStructure().read_image(
            dicom_rtst_path(), structure_name=structure_name, reference_image=image
        )


def test_read_single_dicom_structure_without_reference_image():
    """
    Read a dicom RT Structure without specifying a reference image.

    A value error should be raised.
    """
    structure_name = "GTV-1"
    with pytest.raises(ValueError):
        _ = RTStructure().read_image(dicom_rtst_path(), structure_name=structure_name)


def test_read_single_dicom_structure_without_structure_name():
    """
    Read a dicom RT Structure without specifying a structure name.

    A value error should be raised.
    """
    image = srmip.Image().read_image(dicom_ct_path())
    with pytest.raises(ValueError):
        _ = RTStructure().read_image(dicom_rtst_path(), reference_image=image)


@pytest.mark.parametrize("use_structure_name", [True, False])
@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_read_single_nifti_structure(use_structure_name, extension, tmp_path):
    """Read a nifti RT Structure with or without specifying a structure name."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )
    if use_structure_name is False:
        structure_name = "structure"
    rtst_path = tmp_path / f"{structure_name}.{extension}"
    sitk.WriteImage(rtst, rtst_path)

    rtst = RTStructure().read_image(rtst_path)
    assert rtst.name == structure_name
    # check size
    assert rtst.GetSize() == TEST_RTST_SIZE
    # check spacing
    np.testing.assert_allclose(rtst.GetSpacing(), TEST_RTST_SPACING)


def test_create_structure_set_from_structure():
    """Create a RT Structure Set from a single RT Structure."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst = RTStructureSet([structure])
    assert len(rtst) == 1
    assert list(rtst.keys()) == [structure_name]
    assert list(rtst.values()) == [structure]


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_single_nifti_structure(extension, tmp_path):
    """Create a RT Structure Set from a single RT Structure."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / f"{structure_name}.{extension}"
    structure.write_image(rtst_path)

    saved_structure = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(structure.numpy(), sitk.GetArrayFromImage(saved_structure))


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_nifti_structure_set(extension, tmp_path):
    """Create a RT Structure Set from a single RT Structure."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / f"{structure_name}.{extension}"
    rtst = RTStructureSet([structure])
    rtst.write_image([rtst_path])

    saved_structure = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(structure.numpy(), sitk.GetArrayFromImage(saved_structure))


def test_write_dicom_structure(tmp_path):  # pylint: disable=R0914
    """Create a RT Structure Set from a single RT Structure."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([structure])
    rtst.write_image(rtst_path, reference_image_path=dicom_ct_path())

    original_structure = pydicom.dcmread(dicom_rtst_path())
    saved_structure = pydicom.dcmread(rtst_path)

    comparison_keys = [
        "SOPClassUID",
        "Modality",
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientWeight",
        "StudyInstanceUID",
    ]
    for key in comparison_keys:
        assert original_structure[key] == saved_structure[key]
    for x, y in zip(  # pylint: disable=C0103
        original_structure["ReferencedFrameOfReferenceSequence"],
        saved_structure["ReferencedFrameOfReferenceSequence"],
    ):
        assert x["FrameOfReferenceUID"] == y["FrameOfReferenceUID"]
        for xx, yy in zip(  # pylint: disable=C0103
            x["RTReferencedStudySequence"], y["RTReferencedStudySequence"]
        ):
            assert xx["ReferencedSOPInstanceUID"] == yy["ReferencedSOPInstanceUID"]
            for xxx, yyy in zip(xx["RTReferencedSeriesSequence"], yy["RTReferencedSeriesSequence"]):
                assert xxx["SeriesInstanceUID"] == yyy["SeriesInstanceUID"]
                xxx_ids = {a["ReferencedSOPInstanceUID"].value for a in xxx["ContourImageSequence"]}
                yyy_ids = {a["ReferencedSOPInstanceUID"].value for a in yyy["ContourImageSequence"]}
                assert xxx_ids == yyy_ids
    for x, y in zip(  # pylint: disable=C0103
        original_structure["StructureSetROISequence"], saved_structure["StructureSetROISequence"]
    ):
        assert x["ROINumber"] == y["ROINumber"]
        assert x["ReferencedFrameOfReferenceUID"] == y["ReferencedFrameOfReferenceUID"]
        assert x["ROIName"] == y["ROIName"]

    original_mask = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )
    saved_mask = RTStructure().read_image(
        rtst_path, structure_name=structure_name, reference_image=image
    )
    assert original_mask == saved_mask
    assert np.all(original_mask.numpy() == saved_mask.numpy())


def test_write_dicom_structure_with_hole(tmp_path):
    """Write a structure with a hole inside."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-3"
    reference_structure = RTStructure().read_image(
        dicom_rtst_path_with_hole(), structure_name=structure_name, reference_image=image
    )
    rtst_path = tmp_path / "rtst.dcm"
    reference_structure.write_image(rtst_path, reference_image_path=dicom_ct_path())
    structure = RTStructure().read_image(
        rtst_path, structure_name=structure_name, reference_image=image
    )
    assert structure.numpy().sum() == reference_structure.numpy().sum()
    assert np.all(structure.numpy() == reference_structure.numpy())


def test_write_dicom_structure_set_without_reference(tmp_path):
    """Create a RT Structure Set from a single RT Structure without a reference image."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([structure])
    with pytest.raises(ValueError):
        rtst.write_image(rtst_path)


def test_write_dicom_structure_set(tmp_path):
    """Create a dicom RT Structure Set from a single structure."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / "rtst.dcm"
    structure.write_image(rtst_path, reference_image_path=dicom_ct_path())

    rtst_set_path = tmp_path / "rtst_set.dcm"
    rtst = RTStructureSet([structure])
    rtst.write_image(rtst_set_path, reference_image_path=dicom_ct_path())

    rtst_mask = RTStructure().read_image(
        rtst_path, structure_name=structure_name, reference_image=image
    )
    rtst_set_mask = RTStructure().read_image(
        rtst_set_path, structure_name=structure_name, reference_image=image
    )
    assert rtst_mask == rtst_set_mask


def test_read_dicom_structure_set():
    """Read a dicom rtst file."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructureSet().read_image(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image
    )
    assert len(rtst) == 1
    assert rtst[structure_name].name == structure_name


def test_read_dicom_structure_set_function():
    """Read a dicom rtst file using the function in __init__.py."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = srmip.read_structure_set(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image
    )
    reference_rtst = RTStructureSet().read_image(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image
    )
    assert len(rtst) == len(reference_rtst)
    structure = rtst[structure_name]
    reference_structure = reference_rtst[structure_name]
    assert structure.name == reference_structure.name
    assert structure.GetSize() == reference_structure.GetSize()
    assert structure.spacing == reference_structure.spacing
    assert structure.origin == reference_structure.origin
    assert structure.direction == reference_structure.direction
    assert np.all(structure.numpy() == reference_structure.numpy())


@pytest.mark.parametrize("regex", [True, False])
def test_read_dicom_structure_set_regex(regex):
    """Read a dicom rtst file with a regular expression match."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = r"[A-Z]TV-\d+"
    rtst = RTStructureSet().read_image(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image, regex=regex
    )
    if regex is True:
        assert len(rtst) == 1
        structure_name = "GTV-1"
        assert rtst[structure_name].name == structure_name
    else:
        assert len(rtst) == 0


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_read_nifti_structure_set(extension, tmp_path):
    """Read multiple nifti rtst files."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructureSet().read_image(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image
    )

    rtst_path = tmp_path / f"{structure_name}.{extension}"
    rtst.write_image([rtst_path])

    saved_rtst = RTStructureSet().read_image([rtst_path])
    assert saved_rtst == rtst


def test_read_nifti_structure_set_all_structures():
    """Read all structures from DICOM rtst file."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructureSet().read_image(dicom_rtst_path(), reference_image=image)
    assert tuple(rtst.keys()) == (structure_name,)


def test_dicom_nifti_ibsi_conversion():
    """Test if the DICOM->nifti conversion is IBSI compliant."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    rtst = RTStructureSet().read_image(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=image
    )[structure_name]
    reference_rtst = RTStructure().read_image(ibsi_rtst_path())
    assert np.all(rtst.numpy() == reference_rtst.numpy())


def test_rtstruct_resample():
    """Test RTStructure.resample()."""
    structure = RTStructure.read_image(ibsi_rtst_path())
    resampled_structure = structure.resample((0.8, 0.8, 0.8))
    volume = structure.numpy().sum() * np.prod(structure.spacing)
    resampled_volume = resampled_structure.numpy().sum() * np.prod(resampled_structure.spacing)
    assert np.allclose(resampled_volume, volume, rtol=0.006)


@pytest.mark.parametrize("operation", ["__add__", "__sub__", "__mul__", "__truediv__"])
def test_rtstruct_add_sub_mul_truediv(operation):
    """Arithmetic operators on RT structures are not implemented."""
    structure = RTStructure.read_image(ibsi_rtst_path())
    factor = 0.5
    with pytest.raises(NotImplementedError):
        getattr(structure, operation)(factor)


@pytest.mark.parametrize("set_description", [True, False])
def test_write_dicom_structure_set_description(set_description, tmp_path):
    """Create a dicom RT Structure Set setting the series description."""
    image = srmip.Image().read_image(dicom_ct_path())
    structure_name = "GTV-1"
    structure = RTStructure().read_image(
        dicom_rtst_path(), structure_name=structure_name, reference_image=image
    )

    rtst_path = tmp_path / "rtst.dcm"
    if set_description:
        reference_description = "Structure_Description"
        structure.write_image(
            rtst_path,
            reference_image_path=dicom_ct_path(),
            series_description=reference_description,
        )
    else:
        reference_description = ""
        structure.write_image(rtst_path, reference_image_path=dicom_ct_path())
    series_description = pydicom.dcmread(rtst_path)["SeriesDescription"].value
    assert series_description == reference_description
