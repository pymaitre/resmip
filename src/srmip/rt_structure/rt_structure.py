"""RT Structure class."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import SimpleITK as sitk

from srmip import Image
from srmip.dicom_utils.rtst import read_dicom_rtstruct, write_dicom_rtstruct
from srmip.utils import PathLike

logger = logging.getLogger(__name__)


def get_structure_name_from_filename(filename: Path) -> str:
    """
    Get the structure name from the filename.

    If the file is compressed, e.g.: structure.nii.gz, remove ".nii".
    :param filename: Name of the file.
    :type filename: Path
    :return: Name of the RT Structure.
    :rtype: str
    """
    compress_extensions = [".gz"]
    if filename.suffix in compress_extensions:
        return ".".join(filename.stem.split(".")[:-1])
    return filename.stem


class RTStructure(Image):
    """RT Structure (wrapper of srmip.Image)."""

    _name: str
    """Name of the RT Structure."""

    def __init__(self, *args, name: str = ""):
        """
        Call srmip.Image constructor and set a name for the RT Structure.

        :param name: name of the RT Structure. Defaults to an empty string.
        :type name: str
        """
        if not isinstance(name, str):
            raise ValueError(
                f"type({name}) ({type(name)}) is not a valid type for name. Supported type(s): str."
            )
        super().__init__(*args)
        self._name = name

    @property
    def name(self) -> str:
        """Name of the RT Structure."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @staticmethod
    def read_image(
        filename: PathLike,
        read_metadata: bool = True,
        structure_name: Optional[str] = None,
        reference_image: Optional[Image] = None,
    ) -> RTStructure:
        """
        Read RT Structure from file.

        The image format is automatically determined from filename's suffix.

        :param filename: Name of the file. If filename ends with ".dcm",
            the reader assumes to read a Dicom rtstruct. Otherwise, it assumes a metatadata
            file with the following format exists: f".{filename.stem}.json".
        :type filename: PathLike
        :param read_metadata: If true, read the json file with metadata
            (not applicable for dicom files). Currently not used.
        :type read_metadata: bool
        :param structure_name: Name of the RT Structure (case-sensitive).
            Required for dicom files. Optional for other files (if set to None, use filename).
        :type structure_name: str | None
        :param reference_image: 3D image used as reference for dicom Structures
            (not used for other formats).
        :type reference_image: Image | None
        :return: RT Structure.
        :rtype: RTStructure
        """
        filename = Path(filename)
        if filename.suffix == ".dcm":
            if structure_name is None:
                raise ValueError("Must specify a structure name for dicom RT Structures.")
            if reference_image is None:
                raise ValueError("Must specify a reference image for dicom RT Structures.")
            sitk_image = read_dicom_rtstruct(filename, reference_image, structure_name)[0]
            new_rt_structure = RTStructure(sitk_image.image, name=sitk_image.name)
            return new_rt_structure
        if structure_name is None:
            structure_name = get_structure_name_from_filename(filename)
        new_rt_structure = RTStructure(Image().read_image(filename), name=structure_name)
        return new_rt_structure

    def write_image(
        self,
        filename: PathLike,
        write_metadata: bool = False,
        file_format: Optional[str] = None,
        reference_image_path: Optional[PathLike] = None,
    ) -> None:
        """
        Save RT Structure file.

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        :param filename: Name of the file. If filename is a directory,
            use a the structure's name. For dicom files use the UID.
        :type filename: PathLike
        :param write_metadata: If true, write the json file with metadata
            (not applicable for dicom files). Currently not used.
        :param file_format: Format of the rt structure saved. If None,
            infer it from filename.
        :type file_format: str | None
        :param reference_image_path: Path of the reference dicom image.
            Ignored when saving in formats other than dicom.
        :type reference_image_path: PathLike | None
        """
        # Create an RT Structure Set and save it
        if file_format is None:
            file_format = Path(filename).suffix
        if file_format != ".dcm":
            return self.write_nondicom(filename, file_format)
        return RTStructureSet([self]).write_image(filename, file_format, reference_image_path)

    def write_nondicom(self, filename: PathLike, file_format: Optional[str] = None) -> None:
        """
        Save RT Structure for formats other than dicom.

        :param filename: Name of the file. If filename is a directory,
            use a the structure's name.
        :type filename: PathLike
        :param file_format: Format of the rt structure saved. If None,
            infer it from filename.
        :type file_format: str | None
        """
        filename = Path(filename)
        if filename.is_dir():
            assert file_format is not None
            filename = filename / f"{self.name}{file_format}"
        filename.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self, filename)

    def resample(
        self,
        new_spacing: Iterable,
        interpolator: int = sitk.sitkNearestNeighbor,
        default_pixel_value: float = 0,
    ) -> RTStructure:
        """
        Wrapper of srmip.Image.resample, using the appropriate interpolator.

        Resample the image with a new voxel spacing (in mm).

        :param new_spacing: New voxel spacing of the resampled image (x, y, z) in mm.
        :type new_spacing: Iterable
        :param interpolator: Interpolation method used for image resampling.
            Only nearest neighbors should be used for RT structures.
        :type interpolator: int
        :param default_pixel_value: Default value for pixel intensity.
        :type default_pixel_value: float
        :return: Resampled image.
        :rtype: Image
        """
        if interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "Only sitk.sitkNearestNeighbor should be used when resampling RT structures."
            )
        resampled_image = super().resample(
            new_spacing=new_spacing,
            interpolator=interpolator,
            default_pixel_value=default_pixel_value,
        )
        resampled_structure = RTStructure(resampled_image, name=self.name)
        resampled_structure.metadata = self.metadata
        return resampled_structure


class RTStructureSet(Dict[str, RTStructure]):
    """RT Structure Set (dictionary of [str, RTStructure])."""

    def __init__(self, structures: Optional[List[RTStructure]] = None):
        """Create a dictionary with the given RT Structures."""
        if structures is None:
            structures = []
        self.update({structure.name: structure for structure in structures})

    @staticmethod
    def read_image(
        filename: Union[PathLike, List[PathLike]],
        structure_names: Optional[List[str]] = None,
        regex: bool = False,
        reference_image: Optional[Image] = None,
        parallel: bool = True,
    ) -> RTStructureSet:
        """
        Read RT Structure Set file(s).

        :param filename:
        :param structure_names: Names of the structures to be read.
            Used for reading only specific structures in a dicom files,
            can also be a regular expression.
        :type structure_names: list[str] | None
        :param regex: Whether to consider `structure_names` as a regular expression or not.
        :type regex: bool
        :param parallel: Whether to read structures in parallel or not.
        :type parallel: bool
        :param reference_image: 3D image used as reference for dicom Structures
            (not used for other formats).
        :type reference_image: Image | None
        :return: RT Structure Set.
        :rtype: RTStructureSet
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            structures = read_dicom_rtstruct(
                filename,
                reference_image=reference_image,
                structure_names=structure_names,
                regex=regex,
                parallel=parallel,
            )
            return RTStructureSet(
                [RTStructure(x.image, name=x.name) for x in structures if x.name is not None]
            )
        structures = []
        for f in filename:
            structures.append(RTStructure().read_image(f))
        return RTStructureSet(structures)

    def write_image(
        self,
        filename: Union[PathLike, List[PathLike]],
        file_format: Optional[str] = None,
        reference_image_path: Optional[PathLike] = None,
    ) -> None:
        """
        Save RT Structure Set file(s).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        :param filename: Name of the dicom file.
            For other formats, it is a list of file names with same length of self.
        :type filename: PathLike|list[PathLike]
        :param file_format: Format of the rt structure saved. If None,
            infer it from filename.
        :type file_format: str | None
        :param reference_image_path: Path of the reference dicom image.
            Ignored when saving in formats other than dicom.
        :type reference_image_path: PathLike | None
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            write_dicom_rtstruct(self, filename, reference_image_path)
            return
        filename = [Path(f) for f in filename]
        assert len(filename) == len(self)
        for f in filename:
            assert isinstance(f, Path)
        for structure_filename, structure in zip(filename, self.values()):
            structure.write_nondicom(structure_filename, file_format)
