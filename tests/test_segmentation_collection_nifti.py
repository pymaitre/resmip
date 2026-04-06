"""Test module for segmentation collection objects from/to nifti."""

import numpy as np
import pytest
import SimpleITK as sitk

import resmip
from resmip.segmentation import (
    RTStructureSet,
    SegmentationCollection,
)

from .utils import dicom_rtst_path, dicom_rtst_path_with_hole


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
@pytest.mark.parametrize("class_type", [SegmentationCollection, RTStructureSet])
def test_read_nifti_structure_set(extension, mock_dicom_image: resmip.Image, class_type, tmp_path):
    """Read multiple nifti rtst files."""
    structure_name = "GTV-1"
    rtst = RTStructureSet.read(
        dicom_rtst_path(), structure_names=[structure_name], reference_image=mock_dicom_image
    )

    rtst_path = tmp_path / f"{structure_name}.{extension}"
    rtst.write([rtst_path])

    saved_rtst = class_type.read([rtst_path])
    assert saved_rtst == rtst


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
@pytest.mark.parametrize("class_type", [SegmentationCollection, RTStructureSet])
def test_write_structure_set_to_nifti(
    extension, class_type, mock_dicom_image: resmip.Image, tmp_path
):
    """Create a nifti RT Structure Set/Segmentation."""
    structure_names = ["GTV-1", "GTV-2"]
    rtst = RTStructureSet.read(
        dicom_rtst_path_with_hole(),
        structure_names=structure_names,
        reference_image=mock_dicom_image,
    )
    collection = class_type(rtst.values())

    # save all structures
    save_paths = []
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    collection.write(save_paths, file_format=None)

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            collection[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )

    # save all structures specifying file format
    save_paths = []
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    collection.write(save_paths, file_format=f".{extension}")

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            collection[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )

    # save all structures in directory (not supported)
    with pytest.raises(ValueError):
        collection.write(tmp_path, file_format=None)

    # save all structures in directory specifying file format
    collection.write(tmp_path, file_format=f".{extension}")

    for structure in structure_names:
        saved_structure = sitk.ReadImage(tmp_path / f"{structure}.{extension}")
        np.testing.assert_array_equal(
            collection[structure].numpy(), sitk.GetArrayFromImage(saved_structure)
        )


@pytest.mark.parametrize("extension", ["nii", "nii.gz"])
@pytest.mark.parametrize("class_type", [SegmentationCollection, RTStructureSet])
def test_write_structure_set_to_nifti_wrong_number(
    extension, class_type, mock_dicom_image: resmip.Image, tmp_path
):
    """Create a nifti RT Structure Set/Segmentation providing a wrong number of filenames."""
    structure_names = ["GTV-1"]
    rtst = RTStructureSet.read(
        dicom_rtst_path_with_hole(),
        structure_names=structure_names,
        reference_image=mock_dicom_image,
    )
    collection = class_type(rtst.values())
    save_paths = [
        tmp_path / "GTV-1.nii",
        tmp_path / "GTV-2.nii",
    ]
    for structure in structure_names:
        save_paths.append(tmp_path / f"{structure}.{extension}")
    with pytest.raises(ValueError):
        collection.write(save_paths, file_format=f".{extension}")
