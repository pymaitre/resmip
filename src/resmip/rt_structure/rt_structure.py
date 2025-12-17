"""RT Structure class."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import resmip.dicom_utils.rtst as dicom_rtst
from resmip import Image
from resmip.image._data_types import ImageDTypeLike
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

    def __init__(self, *args, name: str = "", **kwargs):
        """Call resmip.Image constructor and set a name for the RT Structure.

        Args:
            *args: arguments provided to `resmip.Image.__init__`.
            name (str): name of the RT Structure. Defaults to an empty string.
            **kwargs: extra arguments used in `resmip.Image.__init__`.
        """
        if not isinstance(name, str):
            raise ValueError(
                f"type({name}) ({type(name)}) is not a valid type for name. Supported type(s): str."
            )
        super().__init__(*args, **kwargs)
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

    def astype(self, dtype: ImageDTypeLike) -> RTStructure:
        """Convert pixel array type to the specified value, by casting a new structure.

        Args:
            dtype (ImageDTypeLike): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global `FORMAT_TO_TYPESTR` dictionary.

        Returns:
            RTStructure: new structure with specified data type.
        """
        return RTStructure(super().astype(dtype=dtype), name=self.name)

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

        Args:
            filename (PathLike): Name of the file. If filename ends with ".dcm",
                the reader assumes to read a Dicom rtstruct. Otherwise, it assumes a metatadata
                file with the following format exists: f".{filename.stem}.json".
            read_metadata (bool): If true, read the json file with metadata
                (not applicable for dicom files). Currently not used.
            structure_name (str | None): Name of the RT Structure (case-sensitive).
                Required for dicom files. Optional for other files (if set to None, use filename).
            reference_image (Image | None): 3D image used as reference for dicom Structures
                (not used for other formats).

        Returns:
            RTStructure: RT Structure.
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
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        direction: tuple[float],
        metadata: dict[str, str] | None = None,
        name: str = "",
        **kwargs,
    ) -> RTStructure:
        """Create a new structure from a numpy array.

        Args:
            array (np.ndarray): 3D array containing voxel values for the structure (z, y, x).
            spacing (tuple[float, float, float]): Voxel spacing for the structure in mm (x, y, z).
            origin (tuple[float, float, float]): Coordinates of the top left voxel in mm (x, y, z).
            direction (tuple[float]): Direction cosine matrix.
            metadata (dict[str, str] | None): Metadata containing information from the DICOM header.
            name (str): Name of the structure.
            **kwargs: extra arguments used in `resmip.Image.__init__`.

        Returns:
            RTStructure: New structure
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

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                use a the structure's name. For dicom files use the UID.
            write_metadata (bool): If true, write the json file with metadata
                (not applicable for dicom files). Currently not used.
            file_format (str | None): Format of the rt structure saved. If None,
                infer it from filename.
            reference_image_path (PathLike | None): Path of the reference dicom image.
                Ignored when saving in formats other than dicom.
            series_description (str): Series Description for the saved DICOM
                RT Structure Set. Non used for other formats.
        """
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

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                use a the structure's name.
            write_metadata (bool): If true, save the json file with metadata
                (not applicable for dicom files).
            file_format (str | None): Format of the rt structure saved. If None,
                infer it from filename.
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

        Args:
            new_spacing (Iterable): New voxel spacing of the resampled image (x, y, z) in mm.
            interpolator (int): Interpolation method used for image resampling.
                Only nearest neighbors should be used for RT structures.
            default_pixel_value (float): Default value for pixel intensity.

        Returns:
            RTStructure: Resampled structure.
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

        Args:
            reference_image (Image): Image used as reference for padding.
            **kwargs: same arguments used in `np.pad`.

        Returns:
            RTStructure: New structure with same shape and spacing of the reference.
        """
        return RTStructure(super().pad(reference_image=reference_image, **kwargs), name=self.name)

    def coregister(self, *args, **kwargs) -> RTStructure:
        """Coregiser the structure on top of another (reference) image.

        Raises:
            NotImplementedError: Coregistration of structures is not supported.
        """
        raise NotImplementedError

    def __add__(self, value: int | float) -> RTStructure:
        """Add constant value to structure pixel data.

        Args:
            value (int | float): Value to be added to pixel data.

        Returns:
            RTStructure: RTStructure with constant value added to pixel data.
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __sub__(self, value: int | float) -> RTStructure:
        """Subtract constant value to structure pixel data.

        Args:
            value (int | float): Value to be subtracted to pixel data.

        Returns:
            RTStructure: RTStructure with constant value subtracted to pixel data.
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __mul__(self, value: int | float) -> RTStructure:
        """Multiply constant value to structure pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            RTStructure: RTStructure with constant value multiplied to pixel data.
        """
        raise NotImplementedError("This operation is currently not supported for RT Structures.")

    def __truediv__(self, value: int | float) -> RTStructure:
        """Divide constant value to structure pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            RTStructure: RTStructure with constant value multiplied to pixel data.
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

        Args:
            filename (PathLike | list[PathLike]): Name of the DICOM RT structure set.
                If reading from NIfTI, use a list of paths to the structures,
            structure_names (list[str] | None): Names of the structures to be read.
                Used for reading only specific structures in a dicom files,
                can also be a regular expression.
            regex (bool): Whether to consider `structure_names` as a regular expression or not.
            parallel (bool): Whether to read structures in parallel or not.
            reference_image (Image | None): 3D image used as reference for dicom Structures
                (not used for other formats).

        Returns:
            RTStructureSet: RT Structure Set.
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

        Args:
            filename (PathLike | list[PathLike]): Name of the dicom file.
                For other formats, it is a list of file names with same length of self.
            file_format (str | None): Format of the rt structure saved. If None,
                infer it from filename.
            reference_image_path (PathLike | None): Path of the reference dicom image.
                Ignored when saving in formats other than dicom.
            series_description (str): Series Description for the saved DICOM
                RT Structure Set. Non used for other formats.
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
