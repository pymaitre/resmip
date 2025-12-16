"""RT Structure class."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import resmip.dicom_utils.rtst as dicom_rtst
from resmip import Image
from resmip.utils import PathLike

logger = logging.getLogger(__name__)


def get_structure_name_from_filename(filename: Path) -> str:
    """Get the structure name from the filename.

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
    """RT Structure (wrapper of resmip.Image)."""

    def __init__(self, *args, name: str = ""):
        """Call resmip.Image constructor and set a name for the RT Structure.

        :param name: name of the RT Structure. Defaults to an empty string.
        :type name: str
        """
        if not isinstance(name, str):
            raise ValueError(
                f"type({name}) ({type(name)}) is not a valid type for name. Supported type(s): str."
            )
        super().__init__(*args)
        self._name = name

    def __getitem__(self, key) -> RTStructure:
        """Get a pixel value, a sliced image, or a metadata item.

        This operator implements basic indexing where idx is
        arguments or a squence of integers the same dimension as
        the image. The result will be a pixel value from that
        index.

        Multi-dimension extended slice based indexing is also
        implemented. The return is a copy of a new image. The
        standard sliced based indices are supported including
        negative indices, to indicate location relative to the
        end, along with negative step sized to indicate reversing
        of direction.

        If the length of idx is less than the number of dimension
        of the image it will be padded with the defaults slice
        ":".

        When an index element is an integer, that dimension is
        collapsed extracting an image with reduced dimensionality.
        The minimum dimension of an image which can be extracted
        is 2D.

        If indexing with a string, then the metadata dictionary
        queried with the index as the key. If the metadata dictionary
        does not contain the key, a KeyError will occour.
        """
        return RTStructure(super().__getitem__(key), name=self.name)

    @property
    def name(self) -> str:
        """Name of the RT Structure."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @classmethod
    def read_image(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        structure_name: str | None = None,
        reference_image: Image | None = None,
    ) -> RTStructure:
        """Read RT Structure from file.

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
            sitk_image = dicom_rtst.read(filename, reference_image, structure_name)[0]
            new_rt_structure = cls(sitk_image.image, name=sitk_image.name)
            return new_rt_structure
        if structure_name is None:
            structure_name = get_structure_name_from_filename(filename)
        new_rt_structure = cls(Image().read_image(filename), name=structure_name)
        return new_rt_structure

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        spacing: tuple[float],
        origin: tuple[float],
        direction: tuple[float],
        metadata: dict[str, str] | None = None,
        name: str = "",
        **kwargs,
    ) -> RTStructure:
        """Create a new structure from a numpy array.

        :param array: 3D array containing voxel values for the structure (z, y, x).
        :type array: np.ndarray
        :param spacing: Voxel spacing for the structure in mm (x, y, z).
        :type spacing: tuple[float]
        :param origin: Coordinates of the top left voxel in mm (x, y, z).
        :type origin: tuple[float]
        :param direction: Direction cosine matrix.
        :type direction: tuple[float]
        :param metadata: Metadata containing information from the DICOM header.
        :type metadata: Dict[str, str]|None
        :param name: Name of the structure.
        :type name: str
        :return: New structure
        :rtype: RTStructure
        """
        return RTStructure(
            super().from_array(
                array=array,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
                **kwargs,
            ),
            name=name,
        )

    def write_image(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = False,
        file_format: str | None = None,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:
        """Save RT Structure file.

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
        :param series_description: Series Description for the saved DICOM
            RT Structure Set. Non used for other formats.
        :type series_description: str
        """
        # Create an RT Structure Set and save it
        if file_format is None:
            file_format = Path(filename).suffix
        if file_format != ".dcm":
            return self.write_nondicom(filename, file_format=file_format)
        return RTStructureSet([self]).write_image(
            filename,
            file_format=file_format,
            reference_image_path=reference_image_path,
            series_description=series_description,
        )

    def write_nondicom(
        self, filename: PathLike, write_metadata: bool = False, file_format: str | None = None
    ) -> None:
        """Save RT Structure for formats other than dicom.

        :param filename: Name of the file. If filename is a directory,
            use a the structure's name.
        :type filename: PathLike
        :param write_metadata: If true, save the json file with metadata
            (not applicable for dicom files).
        :type write_metadata: bool
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
        """Wrapper of resmip.Image.resample, using the appropriate interpolator.

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
        return resampled_structure

    def pad(self, reference_image: Image, **kwargs) -> RTStructure:
        """Pad the structure on top of another image.

        Uses the same notation as `numpy.pad`.
        The struct is shifted aligning its top-left voxel with the reference image.
        The two images must have the same voxel spacing.
        The shifted structure is cropped if it extends out of the reference image.

        :param reference_image: Image used as reference for padding.
        :type reference_image: Image
        :return: New structure with same shape and spacing of the reference.
        :rtype: RTStructure
        """
        return RTStructure(super().pad(reference_image=reference_image, **kwargs), name=self.name)

    def __add__(self, value: int | float) -> RTStructure:
        """Add constant value to structure pixel data.

        :param value: Value to be added to pixel data.
        :type value: int | float
        :return: RTStructure with constant value added to pixel data.
        :rtype: RTStructure
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __sub__(self, value: int | float) -> RTStructure:
        """Subtract constant value to structure pixel data.

        :param value: Value to be subtracted to pixel data.
        :type value: int | float
        :return: RTStructure with constant value subtracted to pixel data.
        :rtype: RTStructure
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __mul__(self, value: int | float) -> RTStructure:
        """Multiply constant value to structure pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: RTStructure with constant value multiplied to pixel data.
        :rtype: RTStructure
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __truediv__(self, value: int | float) -> RTStructure:
        """Divide constant value to structure pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: RTStructure with constant value multiplied to pixel data.
        :rtype: RTStructure
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")


class RTStructureSet(dict[str, RTStructure]):
    """RT Structure Set (dictionary of [str, RTStructure])."""

    def __init__(self, structures: list[RTStructure] | None = None):
        """Create a dictionary with the given RT Structures."""
        if structures is None:
            structures = []
        self.update({structure.name: structure for structure in structures})

    @classmethod
    def read_image(
        cls,
        filename: PathLike | list[PathLike],
        *,
        structure_names: list[str] | None = None,
        regex: bool = False,
        reference_image: Image | None = None,
        parallel: bool = True,
    ) -> RTStructureSet:
        """Read RT Structure Set file(s).

        :param filename: Name of the DICOM RT structure set.
            If reading from NIfTI, use a list of paths to the structures,
        :type filename: PathLike|list[PathLike]
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
            structures = dicom_rtst.read(
                filename,
                reference_image=reference_image,
                structure_names=structure_names,
                regex=regex,
                parallel=parallel,
            )
            return cls(
                [RTStructure(x.image, name=x.name) for x in structures if x.name is not None]
            )
        structures = []
        for f in filename:
            structures.append(RTStructure().read_image(f))
        return cls(structures)

    def write_image(
        self,
        filename: PathLike | list[PathLike],
        *,
        file_format: str | None = None,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:
        """Save RT Structure Set file(s).

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
        :param series_description: Series Description for the saved DICOM
            RT Structure Set. Non used for other formats.
        :type series_description: str
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            dicom_rtst.write(
                self, filename, reference_image_path, series_description=series_description
            )
            return
        filename = [Path(f) for f in filename]
        assert len(filename) == len(self)
        for f in filename:
            assert isinstance(f, Path)
        for structure_filename, structure in zip(filename, self.values()):
            structure.write_nondicom(structure_filename, file_format)
