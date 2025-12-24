"""Test module for rt_structure.py."""

# pylint: disable=W0621

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

import resmip
from resmip.rt_structure.rt_structure import RTStructure, RTStructureSet

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


def test_read_single_dicom_structure(mock_dicom_image: resmip.Image):
    """Read a dicom RT Structure from file."""
    structure_name = "GTV-1"
    rtst = RTStructure().read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )
    assert rtst.name == structure_name
    # check size
    assert rtst.GetSize() == TEST_RTST_SIZE
    # check spacing
    np.testing.assert_allclose(rtst.GetSpacing(), TEST_RTST_SPACING)


def test_read_single_dicom_structure_function(mock_dicom_image: resmip.Image):
    """Read a dicom RT Structure from file using the function in __init__.py."""
    structure_name = "GTV-1"
    rtst = resmip.read_structure(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )
    reference_rtst = RTStructure.read(
        dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
    )
    assert rtst.name == reference_rtst.name
    assert rtst.GetSize() == reference_rtst.GetSize()
    assert rtst.spacing == reference_rtst.spacing
    assert rtst.origin == reference_rtst.origin
    assert rtst.direction == reference_rtst.direction
    assert np.all(rtst.numpy() == reference_rtst.numpy())


def test_read_single_dicom_structure_wrong_name(mock_dicom_image: resmip.Image):
    """Read a dicom RT Structure from file, with a wrong name."""
    structure_name = "gtv-1"
    with pytest.raises(IndexError):
        _ = RTStructure.read(
            dicom_rtst_path(), structure_name=structure_name, reference_image=mock_dicom_image
        )


def test_read_single_dicom_structure_without_reference_image():
    """Read a dicom RT Structure without specifying a reference image.

    A value error should be raised.
    """
    structure_name = "GTV-1"
    with pytest.raises(ValueError):
        _ = RTStructure.read(dicom_rtst_path(), structure_name=structure_name)


def test_read_single_dicom_structure_without_structure_name(mock_dicom_image: resmip.Image):
    """Read a dicom RT Structure without specifying a structure name.

    A value error should be raised.
    """
    with pytest.raises(ValueError):
        _ = RTStructure.read(dicom_rtst_path(), reference_image=mock_dicom_image)


@pytest.mark.parametrize("use_structure_name", [True, False])
@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_read_single_nifti_structure(
    use_structure_name, extension, mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Read a nifti RT Structure with or without specifying a structure name."""
    structure_name = mock_dicom_structure.name
    if use_structure_name is False:
        structure_name = "structure"
    rtst_path = tmp_path / f"{structure_name}.{extension}"
    sitk.WriteImage(mock_dicom_structure, rtst_path)

    rtst = RTStructure.read(rtst_path)
    assert rtst.name == structure_name
    # check size
    assert rtst.GetSize() == TEST_RTST_SIZE
    # check spacing
    np.testing.assert_allclose(rtst.GetSpacing(), TEST_RTST_SPACING)


def test_create_structure_set_from_structure(mock_dicom_structure: resmip.RTStructure):
    """Create a RT Structure Set from a single RT Structure."""
    rtst = RTStructureSet([mock_dicom_structure])
    assert len(rtst) == 1
    assert list(rtst.keys()) == [mock_dicom_structure.name]
    assert list(rtst.values()) == [mock_dicom_structure]


def test_rtstructure_from_array(mock_dicom_structure: resmip.RTStructure):
    """Test RT structure creation from a numpy array."""
    new_structure = RTStructure.from_array(
        mock_dicom_structure.numpy(),
        spacing=mock_dicom_structure.spacing,
        origin=mock_dicom_structure.origin,
        direction=mock_dicom_structure.direction,
        metadata=mock_dicom_structure.metadata,
        name=mock_dicom_structure.name,
    )
    assert new_structure.GetSize() == mock_dicom_structure.GetSize()
    assert new_structure.spacing == mock_dicom_structure.spacing
    assert new_structure.origin == mock_dicom_structure.origin
    assert new_structure.direction == mock_dicom_structure.direction
    assert np.all(new_structure.numpy() == mock_dicom_structure.numpy())
    assert new_structure.metadata == mock_dicom_structure.metadata
    assert new_structure.name == mock_dicom_structure.name


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_single_nifti_structure(
    extension, mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Create a RT Structure Set from a single RT Structure."""
    rtst_path = tmp_path / f"{mock_dicom_structure.name}.{extension}"
    mock_dicom_structure.write(rtst_path)

    saved_structure = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(
        mock_dicom_structure.numpy(), sitk.GetArrayFromImage(saved_structure)
    )


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_nifti_structure_set(extension, mock_dicom_structure: resmip.RTStructure, tmp_path):
    """Create a RT Structure Set from a single RT Structure."""
    rtst_path = tmp_path / f"{mock_dicom_structure.name}.{extension}"
    rtst = RTStructureSet([mock_dicom_structure])
    rtst.write([rtst_path])

    saved_structure = sitk.ReadImage(rtst_path)
    np.testing.assert_array_equal(
        mock_dicom_structure.numpy(), sitk.GetArrayFromImage(saved_structure)
    )


def test_write_dicom_structure(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure, tmp_path
):  # pylint: disable=R0914
    """Create a RT Structure Set from a single RT Structure."""
    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([mock_dicom_structure])
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

    original_mask = RTStructure.read(
        dicom_rtst_path(),
        structure_name=mock_dicom_structure.name,
        reference_image=mock_dicom_image,
    )
    saved_mask = RTStructure.read(
        rtst_path, structure_name=mock_dicom_structure.name, reference_image=mock_dicom_image
    )
    assert original_mask == saved_mask
    assert np.all(original_mask.numpy() == saved_mask.numpy())


def test_write_dicom_structure_with_hole(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Write a structure with a hole inside."""
    rtst_path = tmp_path / "rtst.dcm"
    mock_dicom_structure.write(rtst_path, reference_image_path=dicom_ct_path())
    structure = RTStructure.read(
        rtst_path, structure_name=mock_dicom_structure.name, reference_image=mock_dicom_image
    )
    assert structure.numpy().sum() == mock_dicom_structure.numpy().sum()
    assert np.all(structure.numpy() == mock_dicom_structure.numpy())


def test_write_dicom_structure_set_without_reference(
    mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Create a RT Structure Set from a single RT Structure without a reference image."""
    rtst_path = tmp_path / "rtst.dcm"
    rtst = RTStructureSet([mock_dicom_structure])
    with pytest.raises(ValueError):
        rtst.write(rtst_path)


def test_write_dicom_structure_set(
    mock_dicom_image: resmip.Image, mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Create a dicom RT Structure Set from a single structure."""
    rtst_path = tmp_path / "rtst.dcm"
    mock_dicom_structure.write(rtst_path, reference_image_path=dicom_ct_path())

    rtst_set_path = tmp_path / "rtst_set.dcm"
    rtst = RTStructureSet([mock_dicom_structure])
    rtst.write(rtst_set_path, reference_image_path=dicom_ct_path())

    rtst_mask = RTStructure.read(
        rtst_path, structure_name=mock_dicom_structure.name, reference_image=mock_dicom_image
    )
    rtst_set_mask = RTStructure.read(
        rtst_set_path, structure_name=mock_dicom_structure.name, reference_image=mock_dicom_image
    )
    assert rtst_mask == rtst_set_mask


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
    assert structure.GetSize() == reference_structure.GetSize()
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


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_read_nifti_structure_set(extension, mock_dicom_image: resmip.Image, tmp_path):
    """Read multiple nifti rtst files."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )

    rtst_path = tmp_path / f"{structure_name}.{extension}"
    rtst.write([rtst_path])

    saved_rtst = RTStructureSet.read([rtst_path])
    assert saved_rtst == rtst


def test_read_nifti_structure_set_all_structures(mock_dicom_image: resmip.Image):
    """Read all structures from DICOM rtst file."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(dicom_rtst_path(), reference_image=mock_dicom_image)
    assert tuple(rtst.keys()) == (structure_name,)


def test_dicom_nifti_ibsi_conversion(mock_dicom_image: resmip.Image):
    """Test if the DICOM->nifti conversion is IBSI compliant."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )[structure_name]
    reference_rtst = RTStructure.read(ibsi_rtst_path())
    assert np.all(rtst.numpy() == reference_rtst.numpy())


def test_rtstruct_resample():
    """Test RTStructure.resample()."""
    structure = RTStructure.read(ibsi_rtst_path())
    resampled_structure = structure.resample((0.8, 0.8, 0.8))
    volume = structure.numpy().sum() * np.prod(structure.spacing)
    resampled_volume = resampled_structure.numpy().sum() * np.prod(resampled_structure.spacing)
    assert np.allclose(resampled_volume, volume, rtol=0.006)


@pytest.mark.parametrize("left_shift", [-1, 0, 1])
@pytest.mark.parametrize("right_shift", [-1, 0, 1])
def test_rtstruct_pad(left_shift, right_shift):
    """Test RTStructure.pad()."""
    structure_spacing = (1, 1, 1)
    structure_origin = np.array((0, 0, 0))
    reference_origin = structure_origin + left_shift
    structure_direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    original_structure_shape = (5, 5, 5)
    reference_size = np.array(original_structure_shape) + right_shift
    original_array = np.zeros(original_structure_shape)
    point_coordinate = (2, 2, 2)
    original_array[point_coordinate] = 1
    original_structure = RTStructure.from_array(
        original_array,
        spacing=structure_spacing,
        origin=tuple(structure_origin.tolist()),
        direction=structure_direction,
        name="Struct",
    )
    reference_structure = RTStructure.from_array(
        np.zeros(reference_size),
        spacing=structure_spacing,
        origin=tuple(reference_origin.tolist()),
        direction=structure_direction,
        name="Struct",
    )
    padded_structure = original_structure.pad(reference_structure)
    new_coordinate = np.array(point_coordinate) - left_shift

    assert padded_structure.numpy().shape == reference_structure.numpy().shape
    assert padded_structure.numpy()[tuple(new_coordinate.tolist())] == 1
    assert padded_structure.name == reference_structure.name


@pytest.mark.parametrize("set_description", [True, False])
def test_write_dicom_structure_set_description(
    set_description, mock_dicom_structure: resmip.RTStructure, tmp_path
):
    """Create a dicom RT Structure Set setting the series description."""
    rtst_path = tmp_path / "rtst.dcm"
    if set_description:
        reference_description = "Structure_Description"
        mock_dicom_structure.write(
            rtst_path,
            reference_image_path=dicom_ct_path(),
            series_description=reference_description,
        )
    else:
        reference_description = ""
        mock_dicom_structure.write(rtst_path, reference_image_path=dicom_ct_path())
    series_description = pydicom.dcmread(rtst_path)["SeriesDescription"].value
    assert series_description == reference_description


def test_crop_structure(mock_dicom_structure: resmip.RTStructure):
    """Crop RT structure."""
    cropped_structure = mock_dicom_structure[1:-2, 1:-2, 1:-2]
    padded_structure = cropped_structure.pad(mock_dicom_structure)

    np.testing.assert_equal(
        np.asarray(cropped_structure.GetSize()),
        np.asarray(mock_dicom_structure.GetSize()) - 3,
    )
    assert padded_structure.GetSize() == mock_dicom_structure.GetSize()
    assert padded_structure.origin == mock_dicom_structure.origin
    assert padded_structure.spacing == mock_dicom_structure.spacing
    np.testing.assert_equal(
        padded_structure.numpy(),
        mock_dicom_structure.numpy(),
    )


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_structure_set_to_nifti(extension, mock_dicom_image: resmip.Image, tmp_path):
    """Create a nifti RT Structure Set."""
    structure_names = ["GTV-1", "GTV-2"]
    rtst = RTStructureSet.read(
        dicom_rtst_path_with_hole(),
        structure_names=structure_names,
        reference_image=mock_dicom_image,
    )

    # save all structures
    save_paths = []
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    rtst.write(save_paths, file_format=None)

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            rtst[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )

    # save all structures specifying file format
    save_paths = []
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    rtst.write(save_paths, file_format=f".{extension}")

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            rtst[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )

    # save all structures in directory (not supported)
    with pytest.raises(ValueError):
        rtst.write(tmp_path, file_format=None)

    # save all structures in directory specifying file format
    rtst.write(tmp_path, file_format=f".{extension}")

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            rtst[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
def test_write_structure_set_to_nifti_wrong_number(
    extension, mock_dicom_image: resmip.Image, tmp_path
):
    """Create a nifti RT Structure Set providing a wrong number of filenames."""
    structure_names = ["GTV-1"]
    rtst = RTStructureSet.read(
        dicom_rtst_path_with_hole(),
        structure_names=structure_names,
        reference_image=mock_dicom_image,
    )
    save_paths = [
        tmp_path / "GTV-1.nii",
        tmp_path / "GTV-2.nii",
    ]
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    with pytest.raises(ValueError):
        rtst.write(save_paths, file_format=f".{extension}")


@pytest.mark.parametrize("structure_path", [dicom_rtst_path, ibsi_rtst_path])
@pytest.mark.parametrize("read_metadata", [True, False])
def test_structure_modality(structure_path, read_metadata):
    """Test if RT structure modality is displayed correctly."""
    structure_path = structure_path()
    structure_name = "GTV-1"
    if structure_path.suffix == ".dcm":
        image = resmip.Image.read(dicom_ct_path())
        structure = RTStructure.read(
            structure_path, structure_name=structure_name, reference_image=image
        )
    else:
        structure = RTStructure.read(structure_path, read_metadata=read_metadata)
    assert structure.name == structure_name
    assert structure.modality == "RTSTRUCT"
