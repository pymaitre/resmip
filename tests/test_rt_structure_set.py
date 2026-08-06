"""Test module for rt_structure.py."""

# pylint: disable=W0621

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from dicom_validator.validator.validation_result import DicomTag, ErrorCode

import resmip
from resmip.image import DicomModality
from resmip.segmentation import RTStructureSet, Segmentation

from .utils import (
    dicom_ct_path,
    dicom_rtst_path,
    ibsi_rtst_path,
)

TEST_RTST_SIZE = (204, 201, 60)
"""Size of the test RT Structure."""
TEST_RTST_SPACING = (0.97699999809265, 0.97699999809265, 2.9999999999998486)
"""Spacing of the test RT Structure."""


def assert_required_tags_in_dicom_rtstruct(dataset: pydicom.Dataset):
    """Check if all required fields for DICOM RTSTRUCT are present."""
    assert dataset.file_meta.MediaStorageSOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"
    assert dataset.file_meta.TransferSyntaxUID is not None

    assert dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.3"
    assert dataset.SOPInstanceUID is not None
    assert dataset.Modality == "RTSTRUCT"
    for mandatory_uid in ("SeriesInstanceUID", "StudyInstanceUID", "PatientID"):
        assert hasattr(dataset, mandatory_uid)


def test_read_dicom_structure_set(mock_dicom_image: resmip.Image):
    """Read a dicom rtst file."""
    structure_name = "GTV-1"
    rtst = RTStructureSet().read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )
    assert len(rtst) == 1
    assert rtst[structure_name].name == structure_name


def test_read_dicom_structure_set_function(mock_dicom_image: resmip.Image):
    """Read a dicom rtst file using the function in __init__.py."""
    structure_name = "GTV-1"
    rtst = resmip.read_structure_set(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )
    reference_rtst = RTStructureSet.read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )
    assert len(rtst) == len(reference_rtst)
    structure = rtst[structure_name]
    reference_structure = reference_rtst[structure_name]
    assert structure.name == reference_structure.name
    assert structure.size == reference_structure.size
    assert structure.spacing == reference_structure.spacing
    assert structure.origin == reference_structure.origin
    assert structure.direction == reference_structure.direction
    assert np.all(structure.numpy() == reference_structure.numpy())


@pytest.mark.parametrize("regex", [True, False])
def test_read_dicom_structure_set_regex(regex, mock_dicom_image: resmip.Image):
    """Read a dicom rtst file with a regular expression match."""
    structure_name = r"[A-Z]TV-\d+"
    rtst = RTStructureSet.read(
        dicom_rtst_path(),
        structure_names=[structure_name],
        reference_image=mock_dicom_image,
        regex=regex,
    )
    if regex is True:
        assert len(rtst) == 1
        structure_name = "GTV-1"
        assert rtst[structure_name].name == structure_name
    else:
        assert len(rtst) == 0


def test_create_structure_set_from_segmentation(mock_dicom_segmentation: Segmentation):
    """Create a RT Structure Set from a single Segmentation."""
    rtst = RTStructureSet([mock_dicom_segmentation])
    assert len(rtst) == 1
    assert list(rtst.keys()) == [mock_dicom_segmentation.name]
    assert list(rtst.values()) == [mock_dicom_segmentation]
    assert rtst.modality == DicomModality.rtstruct.value


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_nifti_structure_set(extension, mock_dicom_segmentation: Segmentation, tmp_path):
    """Create a RT Structure Set from a single Segmentation."""
    rtst_path = tmp_path / f"{mock_dicom_segmentation.name}.{extension}"
    rtst = RTStructureSet([mock_dicom_segmentation])
    rtst.write([rtst_path])

    saved_structure = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(
        mock_dicom_segmentation.numpy(), sitk.GetArrayFromImage(saved_structure)
    )


def test_write_dicom_structure_set(
    mock_dicom_image: resmip.Image, mock_dicom_segmentation: Segmentation, tmp_path
):
    """Create a dicom RT Structure Set from a single segmentation."""
    rtst_path = tmp_path / "rtst.dcm"
    mock_dicom_segmentation.write(
        rtst_path, modality=DicomModality.rtstruct, reference_image_path=dicom_ct_path()
    )

    rtst_set_path = tmp_path / "rtst_set.dcm"
    rtst = RTStructureSet([mock_dicom_segmentation])
    rtst.write(rtst_set_path, reference_image_path=dicom_ct_path())

    rtst_mask = Segmentation.read(
        rtst_path, structure_name=mock_dicom_segmentation.name, reference_image=mock_dicom_image
    )
    rtst_set_mask = Segmentation.read(
        rtst_set_path, structure_name=mock_dicom_segmentation.name, reference_image=mock_dicom_image
    )
    assert rtst_mask == rtst_set_mask
    assert pydicom.dcmread(rtst_path).Modality == "RTSTRUCT"
    assert pydicom.dcmread(rtst_set_path).Modality == "RTSTRUCT"


def test_write_dicom_segmentation_rtst(
    mock_dicom_image: resmip.Image, mock_dicom_segmentation: Segmentation, tmp_path
):  # pylint: disable=R0914
    """Create a RT Structure Set from a single Segmentation."""
    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([mock_dicom_segmentation])
    rtst.write(rtst_path, reference_image_path=dicom_ct_path())

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

    original_mask = Segmentation.read(
        dicom_rtst_path(),
        structure_name=mock_dicom_segmentation.name,
        reference_image=mock_dicom_image,
    )
    saved_mask = Segmentation.read(
        rtst_path, structure_name=mock_dicom_segmentation.name, reference_image=mock_dicom_image
    )
    assert original_mask == saved_mask
    assert np.all(original_mask.numpy() == saved_mask.numpy())


def test_write_dicom_structure_with_hole(
    mock_dicom_image: resmip.Image, mock_dicom_segmentation: Segmentation, tmp_path
):
    """Write a structure with a hole inside."""
    rtst_path = tmp_path / "rtst.dcm"
    RTStructureSet([mock_dicom_segmentation]).write(rtst_path, reference_image_path=dicom_ct_path())
    structure = Segmentation.read(
        rtst_path, structure_name=mock_dicom_segmentation.name, reference_image=mock_dicom_image
    )
    assert structure.numpy().sum() == mock_dicom_segmentation.numpy().sum()
    assert np.all(structure.numpy() == mock_dicom_segmentation.numpy())


def test_write_dicom_structure_set_without_reference(
    mock_dicom_segmentation: Segmentation, tmp_path
):
    """Create a RT Structure Set from a single RT Structure without a reference image."""
    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([mock_dicom_segmentation])
    with pytest.raises(ValueError):
        rtst.write(rtst_path)


@pytest.mark.parametrize("set_description", [True, False])
def test_write_dicom_structure_set_description(
    set_description, mock_dicom_segmentation: Segmentation, tmp_path
):
    """Create a dicom RT Structure Set setting the series description."""
    rtst_path = tmp_path / "rtst.dcm"
    if set_description:
        reference_description = "Structure_Description"
        mock_dicom_segmentation.write(
            rtst_path,
            modality=DicomModality.rtstruct.value,
            reference_image_path=dicom_ct_path(),
            series_description=reference_description,
        )
    else:
        reference_description = ""
        mock_dicom_segmentation.write(
            rtst_path, modality=DicomModality.rtstruct.value, reference_image_path=dicom_ct_path()
        )
    saved_dataset = pydicom.dcmread(rtst_path)
    series_description = saved_dataset["SeriesDescription"].value
    assert series_description == reference_description
    assert_required_tags_in_dicom_rtstruct(saved_dataset)


def test_read_dicom_structure_set_all_structures(mock_dicom_image: resmip.Image):
    """Read all structures from DICOM rtst file."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(dicom_rtst_path(), reference_image=mock_dicom_image)
    assert tuple(rtst.keys()) == (structure_name,)


def test_dicom_nifti_ibsi_conversion(mock_dicom_image: resmip.Image):
    """Test if the DICOM->nifti conversion is IBSI compliant for RT Structures."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )[structure_name]
    reference_rtst = Segmentation.read(ibsi_rtst_path())
    assert np.all(rtst.numpy() == reference_rtst.numpy())


def test_written_dicom_rtstruct_is_dicom_conformant(
    mock_dicom_segmentation: Segmentation, dicom_validator, tmp_path
):
    """Check that the written DICOM RTSTRUCT follows DICOM standard."""
    output_path = tmp_path / "seg.dcm"
    mock_dicom_segmentation.write(
        output_path, modality=DicomModality.rtstruct.value, reference_image_path=dicom_ct_path()
    )

    result = next(iter(dicom_validator.validate(output_path).values()))

    assert result.errors == 2, result.module_errors
    assert len(result.module_errors["ROI Contour"]) == 1
    assert (
        result.module_errors["ROI Contour"][
            DicomTag(tag=0x30060050, parents=[0x30060039, 0x30060040])
        ].code
        == ErrorCode.InvalidValue
    )
    assert len(result.module_errors["RT ROI Observations"]) == 1
    assert (
        result.module_errors["RT ROI Observations"][
            DicomTag(tag=0x30060085, parents=[0x30060080])
        ].code
        == ErrorCode.TagUnexpected
    )


def test_written_new_dicom_rtstruct_is_dicom_conformant(
    mock_dicom_segmentation: Segmentation, dicom_validator, tmp_path
):
    """Check that the newly written DICOM RTSTRUCT follows DICOM standard."""
    output_path = tmp_path / "seg.dcm"
    new_array = np.zeros_like(mock_dicom_segmentation.numpy(copy=False))
    new_array[1, 2:8, 3:5] = 1
    new_segmentation = Segmentation.from_array(
        new_array,
        spacing=mock_dicom_segmentation.spacing,
        origin=mock_dicom_segmentation.origin,
        direction=mock_dicom_segmentation.direction,
        name=mock_dicom_segmentation.name,
        segmentation_type=mock_dicom_segmentation.segmentation_type,
    )
    new_segmentation.write(
        output_path, modality=DicomModality.rtstruct.value, reference_image_path=dicom_ct_path()
    )

    result = next(iter(dicom_validator.validate(output_path).values()))

    assert result.errors == 2, result.module_errors
    assert len(result.module_errors["ROI Contour"]) == 1
    assert (
        result.module_errors["ROI Contour"][
            DicomTag(tag=0x30060050, parents=[0x30060039, 0x30060040])
        ].code
        == ErrorCode.InvalidValue
    )
    assert len(result.module_errors["RT ROI Observations"]) == 1
    assert (
        result.module_errors["RT ROI Observations"][
            DicomTag(tag=0x30060085, parents=[0x30060080])
        ].code
        == ErrorCode.TagUnexpected
    )
