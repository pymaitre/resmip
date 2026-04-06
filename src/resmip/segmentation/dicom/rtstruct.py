"""DICOM RTSTRUCT modality."""

from __future__ import annotations

import logging
from functools import partial
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors
import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.valuerep import IS
from skimage.draw import polygon

from .annotation import (
    DicomAnnotation,
    _duplicate_annotation_names,
    _match_annotation_names,
)
from .rt_utils_wrapper import RTStruct

if TYPE_CHECKING:
    from resmip import Image
    from resmip.segmentation.segmentation import RTStructureSet

logger = logging.getLogger(__name__)


def _check_if_valid_structure(
    struct_index: int | IS,
    struct_point_sequence: dict[str, pydicom.Dataset],
) -> bool:
    """Check if the structure point sequence is valid.

    If the structure is invalid, print more information and return false.

    Args:
        struct_index (int | IS): ROI Number of the RT Structure.
        struct_point_sequence (Dict[str, Dataset]): dictionary containing
            the sequence of points of the RT Structure.

                - key: string representing ROI Number
                - value: Dataset containing RT Structure data (including slice polygons)

    Returns:
        bool: ``True`` if the structure point sequence is valid.
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


def _convert_single_structure(
    reference_image: Image,
    struct_point_sequence: dict[str, pydicom.Dataset],
    struct_ds: pydicom.Dataset,
) -> DicomAnnotation:
    """Convert a DICOM RT Structure to NIFTI.

    Args:
        reference_image (Image): 3D image associated with the structure.
        struct_point_sequence (dict[str, Dataset]): dictionary containing the
            sequence of points of the RT Structure.

                - key: string representing ROI Number
                - value: Dataset containing RT Structure data (including slice polygons)
        struct_ds (Dataset): single element of the Structure Set ROI Sequence
            containing ROI information, including ROI Number and ROI Name.

    Returns:
        DicomStructure: object containing structure name and structure image.
        If the contour is not valid, return an empy DicomStructure(None, None)
    """
    image_blank = np.zeros(reference_image.size[::-1], dtype=np.uint8)

    # struct_name = "_".join(struct_ds.ROIName.split())
    struct_name = str(struct_ds.ROIName)
    struct_index = struct_ds.ROINumber
    # logger.debug("Converting structure %s with name: %s", struct_index, struct_name)

    is_valid_structure = _check_if_valid_structure(struct_index, struct_point_sequence)
    if is_valid_structure is False:
        return DicomAnnotation(None, None)

    # Track in case something goes wrong in here we will skip the contour
    skip_contour = False
    for sl, _ in enumerate(struct_point_sequence[struct_index].ContourSequence):
        contour_data = np.array(
            struct_point_sequence[struct_index].ContourSequence[sl].ContourData, dtype=float
        ).reshape(-1, 3)

        contour_vertices = (contour_data - reference_image.origin) / reference_image.spacing

        z_index = contour_vertices[0, 2]
        if np.any(contour_vertices[:, 2] != z_index):
            logger.debug("Error: axial slice index varies in contour. Skipping Contour.")
            logger.debug("Structure:   %s", struct_name)
            logger.debug("Slice index: %d", z_index)
            skip_contour = True
            break

        if z_index >= reference_image.size[2] or z_index < 0:
            logger.debug(
                "Warning: Slice index greater than image size or less than zero. Skipping slice."
            )
            logger.debug("Structure:   %s", struct_name)
            logger.debug("Slice index: %d", z_index)
            continue
        z_index = int(z_index.round())

        filled_indices_x, filled_indices_y = polygon(
            contour_vertices[:, 0], contour_vertices[:, 1], shape=image_blank.shape[1:]
        )
        image_blank[z_index, filled_indices_y, filled_indices_x] = 1

    if not skip_contour:
        struct_image = sitk.GetImageFromArray(1 * (image_blank > 0))
        struct_image.CopyInformation(reference_image)
        return DicomAnnotation(struct_name, struct_image)
    return DicomAnnotation(None, None)


def _read(  # pylint: disable=too-many-locals
    rtst_path: Path,
    reference_image: Image,
    structure_names: str | list[str] | None = None,
    parallel: bool = False,
    regex: bool = False,
    case_sensitive: bool = True,
) -> list[DicomAnnotation]:
    """Read DICOM ST Structure Set file and convert it into a list of RTStructure (nifti) objects.

    Args:
        rtst_path (Path): full path of the RT Structure Set.
        reference_image (Image): 3D image associated with the structure set.
        structure_names (str | list[str] | None): structure name or list
            of structure names to convert. Other structures will not be converted.
            If set to None, all structures found will be converted.
        parallel (bool): read RT Structures in parallel.
        regex (bool): if set to true, structure names are searched as regular expression pattern,
            otherwise only exact matches are returned.

    Returns:
        list[DicomAnnotation]: list of matching RTStructure (nifti) objects.
    """
    dataset = pydicom.dcmread(rtst_path, force=True)

    struct_point_sequence = {cs.ReferencedROINumber: cs for cs in dataset.ROIContourSequence}
    structure_sets = []

    dicom_structures = dataset.StructureSetROISequence
    if _duplicate_annotation_names(dicom_structures, key="ROIName"):
        raise ValueError("Multiple structures with the same name were found.")

    if isinstance(structure_names, str):
        structure_names = [structure_names]

    if structure_names is not None:
        matched_dicom_structures = _match_annotation_names(
            annotation_sequence=dicom_structures,
            names_to_match=structure_names,
            key="ROIName",
            regex=regex,
            case_sensitive=case_sensitive,
        )
    else:
        matched_dicom_structures = dicom_structures
    if len(matched_dicom_structures) == 0:
        logger.warning("No matching structures found.")
        return []
    if parallel is True:
        num_threads = max(len(matched_dicom_structures), 1)
        with ThreadPool(num_threads) as p:
            structure_sets = p.map(
                partial(_convert_single_structure, reference_image, struct_point_sequence),
                matched_dicom_structures,
            )
    else:
        for matched_dicom_structure in matched_dicom_structures:
            nifti_structure = _convert_single_structure(
                reference_image, struct_point_sequence, matched_dicom_structure
            )
            structure_sets.append(nifti_structure)
    return [struct for struct in structure_sets if struct.name is not None]


def _write(
    rt_structures: RTStructureSet,
    save_path: Path,
    dcm_series_path: Path,
    color_map: matplotlib.colors.Colormap = matplotlib.colormaps.get_cmap("rainbow"),
    series_description: str = "",
) -> None:
    """Write RT Structures to dicom file.

    Wrapper of convert_nifti from platipy.dicom.io.nifti_to_rtstruct.

    Args:
        rt_structures (RTStructureSet): collection of structure name and structure mask.
        save_path (Path): full path of the generated dicom file.
        dcm_series_path (Path): path of the directory containing the reference dicom image.
        color_map (matplotlib.colors.Colormap): Colormap to use for output. Defaults to
            matplotlib.colormaps.get_cmap("rainbow").
        series_description (str): Series Description for the saved DICOM
            RT Structure Set.
    """
    if dcm_series_path is None:
        raise ValueError("The path of the reference dicom series must be specified.")

    rtstruct = RTStruct.create_new(
        dicom_series_path=str(dcm_series_path), series_description=series_description
    )

    for name, structure in rt_structures.items():
        # Use a hash of the name to get the color from the supplied color map
        color = color_map(hash(name) % 256)
        color = color[:3]
        color = [int(c * 255) for c in color]

        rtstruct.add_roi(
            mask=structure,
            color=color,
            name=name,
        )

    rtstruct.save(save_path)
