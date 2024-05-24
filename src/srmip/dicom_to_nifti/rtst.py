"""Read and write dicom RT Structure Sets."""

import logging
import re
from dataclasses import dataclass
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib
import numpy as np
import pydicom as pydcm
import SimpleITK as sitk
from platipy.dicom.io import rtstruct_to_nifti
from rt_utils import RTStructBuilder
from skimage.draw import polygon

from srmip.utils import PathLike

logger = logging.getLogger(__name__)


@dataclass
class DicomStructure:
    """Structure read by the platipy wrapper."""

    name: str
    """Name of the structure."""
    image: sitk.Image
    """Structure mask as SimpleITK Image."""


def check_if_valid_structure(
    struct_index: Union[int, pydcm.valuerep.IS],
    struct_point_sequence: Dict[str, pydcm.dataset.Dataset],
) -> bool:
    """
    Check if the structure point sequence is valid.

    If the structure is invalid, print more information and return false.

    :param struct_index: ROI Number of the RT Structure.
    :type struct_index: int|pydcm.valuerep.IS
    :param struct_point_sequence: dictionary containing the sequence of points of the RT Structure.
        - key: string representing ROI Number
        - value: pydcm.dataset.Dataset containing RT Structure data (including slice polygons)
    :type struct_point_sequence: Dict[str, pydcm.dataset.Dataset]
    :return: true if the structure point sequence is valid.
    :rtype: bool
    """
    if struct_index not in struct_point_sequence:
        logger.debug("No ROIContourSequence found for this structure, skipping.")
        return False
    if not hasattr(struct_point_sequence[struct_index], "ContourSequence"):
        logger.debug("No ContourSequence found for this structure, skipping.")
        return False
    if len(struct_point_sequence[struct_index].ContourSequence) == 0:
        logger.debug("Contour sequence empty for this structure, skipping.")
        return False
    if (
        not struct_point_sequence[struct_index].ContourSequence[0].ContourGeometricType
        == "CLOSED_PLANAR"
    ):
        logger.debug("This is not a closed planar structure, skipping.")
        return False
    return True


def convert_single_structure(  # pylint: disable=too-many-locals
    reference_image: sitk.Image,
    struct_point_sequence: Dict[str, pydcm.dataset.Dataset],
    struct_ds: pydcm.dataset.Dataset,
) -> DicomStructure:
    """
    Convert a DICOM RT Structure to NIFTI.

    :param reference_image: 3D image associated with the structure.
    :type reference_image: sitk.Image
    :param struct_point_sequence: dictionary containing the sequence of points of the RT Structure.
        - key: string representing ROI Number
        - value: pydcm.dataset.Dataset containing RT Structure data (including slice polygons)
    :type struct_point_sequence: Dict[str, pydcm.dataset.Dataset]
    :param struct_ds: single element of the Structure Set ROI Sequence containing ROI information,
        including ROI Number and ROI Name.
    :type struct_ds: pydcm.dataset.Dataset
    :return: object containing structure name and structure image.
        If the contour is not valid, return an empy DicomStructure(None, None)
    :rtype: DicomStructure
    """
    image_blank = np.zeros(reference_image.GetSize()[::-1], dtype=np.uint8)

    struct_name = "_".join(struct_ds.ROIName.split())
    struct_index = struct_ds.ROINumber
    # logger.debug("Converting structure %s with name: %s", struct_index, struct_name)

    is_valid_structure = check_if_valid_structure(struct_index, struct_point_sequence)
    if is_valid_structure is False:
        return DicomStructure(None, None)

    # Track in case something goes wrong in here we will skip the contour
    skip_contour = False
    for sl in range(  # pylint: disable=consider-using-enumerate
        len(struct_point_sequence[struct_index].ContourSequence)
    ):
        contour_data = rtstruct_to_nifti.fix_missing_data(
            struct_point_sequence[struct_index].ContourSequence[sl].ContourData
        )

        struct_slice_contour_data = np.array(contour_data, dtype=np.double)
        vertex_arr_physical = struct_slice_contour_data.reshape(
            struct_slice_contour_data.shape[0] // 3, 3
        )

        point_arr = np.array(
            [reference_image.TransformPhysicalPointToIndex(i) for i in vertex_arr_physical]
        ).T

        [x_vertex_arr_image, y_vertex_arr_image] = point_arr[[0, 1]]
        z_index = point_arr[2][0]
        if np.any(point_arr[2] != z_index):
            logger.debug("Error: axial slice index varies in contour. Skipping Contour.")
            logger.debug("Structure:   %s", struct_name)
            logger.debug("Slice index: %d", z_index)
            skip_contour = True
            break

        if z_index >= reference_image.GetSize()[2]:
            logger.debug("Warning: Slice index greater than image size. Skipping slice.")
            logger.debug("Structure:   %s", struct_name)
            logger.debug("Slice index: %d", z_index)
            continue

        slice_arr = np.zeros(image_blank.shape[-2:], dtype=np.uint8)

        filled_indices_x, filled_indices_y = polygon(
            x_vertex_arr_image, y_vertex_arr_image, shape=slice_arr.shape
        )
        slice_arr[filled_indices_y, filled_indices_x] = 1

        image_blank[z_index] += slice_arr

    if not skip_contour:
        struct_image = sitk.GetImageFromArray(1 * (image_blank > 0))
        struct_image.CopyInformation(reference_image)
        return DicomStructure(struct_name, struct_image)
    return DicomStructure(None, None)


def read_dicom_rtstruct(  # pylint: disable=too-many-locals
    rtst_path: Path,
    reference_image: sitk.Image,
    structure_names: Union[str, list[str]] = None,
    spacing_override: Union[Tuple[float], list[float]] = None,
    parallel: bool = False,
    regex: bool = False,
) -> list[DicomStructure]:
    """
    Read DICOM ST Structure Set file and convert it into a list of RTStructure (nifti) objects.

    :param rtst_path: full path of the RT Structure Set.
    :type rtst_path: Path
    :param reference_image: 3D image associated with the structure set.
    :type reference_image: sitk.Image
    :param structure_names: structure name or list of structure names to convert.
        Other structures will not be converted.
        If set to None, all structures found will be converted.
    :type structure_names: str|list[str]
    :param spacing_override: The spacing to override. Defaults to None.
    :type spacing_override: Tuple[float]|list[float]]
    :param parallel: read RT Structures in parallel.
    :type parallel: bool
    :param regex: if set to true, structure names are searched as regular expression pattern,
        otherwise only exact matches are returned.
    :type regex: bool
    :return: list of matching RTStructure (nifti) objects.
    :rtype: list[DicomStructure]
    """
    # if rtst_path.is_dir():
    #     rtst_path = list(rtst_path.iterdir())[0]
    dicom_struct = pydcm.dcmread(rtst_path, force=True)

    if spacing_override:
        current_spacing = list(reference_image.GetSpacing())
        new_spacing = tuple(  # pylint: disable=consider-using-generator
            [
                current_spacing[k] if spacing_override[k] == 0 else spacing_override[k]
                for k in range(3)
            ]
        )
        reference_image.SetSpacing(new_spacing)

    struct_point_sequence = {cs.ReferencedROINumber: cs for cs in dicom_struct.ROIContourSequence}
    structure_sets = []
    dicom_structures = list(dicom_struct["StructureSetROISequence"].value)

    if isinstance(structure_names, str):
        structure_names = [structure_names]

    if structure_names is not None:
        matched_dicom_structures = []
        for dicom_structure in dicom_structures:
            if regex is True:
                for structure_name in structure_names:
                    if re.search(structure_name.lower(), dicom_structure["ROIName"].value.lower()):
                        matched_dicom_structures.append(dicom_structure)
                        break
            else:
                if dicom_structure["ROIName"].value in structure_names:
                    matched_dicom_structures.append(dicom_structure)
    else:
        matched_dicom_structures = dicom_structures
    if len(matched_dicom_structures) == 0:
        logger.warning("Not matching structures found.")
    if parallel is True:
        num_threads = max(len(matched_dicom_structures), 1)
        with ThreadPool(num_threads) as p:
            structure_sets = p.map(
                partial(convert_single_structure, reference_image, struct_point_sequence),
                matched_dicom_structures,
            )
    else:
        for matched_dicom_structure in matched_dicom_structures:
            nifti_structure = convert_single_structure(
                reference_image, struct_point_sequence, matched_dicom_structure
            )
            structure_sets.append(nifti_structure)
    return structure_sets


def write_dicom_rtstruct(
    rt_structures: dict[str, sitk.Image],
    save_path: PathLike,
    dcm_series_path: PathLike,
    color_map: matplotlib.colors.Colormap = matplotlib.colormaps.get_cmap("rainbow"),
) -> None:
    """
    Write RT Structures to dicom file.

    Wrapper of convert_nifti from platipy.dicom.io.nifti_to_rtstruct.
    :param rt_structures: collection of structure name and structure mask.
    :type rt_structures: dict[str, sitk.Image]
    :param save_path: full path of the generated dicom file.
    :type save_path: PathLike
    :param save_path: path of the directory containing the reference dicom image.
    :type save_path: PathLike
    :param color_map: Colormap to use for output. Defaults to
            matplotlib.colormaps.get_cmap("rainbow").
    :type color_map: matplotlib.colors.Colormap
    """
    logger.info("Will convert the following masks to RTStruct:")
    save_path = Path(save_path)
    if dcm_series_path is None:
        raise ValueError("The path of the reference dicom series must be specified.")
    dcm_series_path = Path(dcm_series_path)

    rtstruct = RTStructBuilder.create_new(dicom_series_path=str(dcm_series_path))

    for mask_name in rt_structures:
        # Use a hash of the name to get the color from the supplied color map
        color = color_map(hash(mask_name) % 256)
        color = color[:3]
        color = [int(c * 255) for c in color]

        mask = rt_structures[mask_name]
        if not isinstance(mask, sitk.Image):
            mask = sitk.ReadImage(str(mask))

        bool_arr = sitk.GetArrayFromImage(mask) != 0
        bool_arr = np.transpose(bool_arr, (1, 2, 0))
        rtstruct.add_roi(mask=bool_arr, color=color, name=mask_name)

    rtstruct.save(str(save_path))
