"""RT Structure class."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import resmip.dicom_utils.rtst as dicom_rtst
from resmip import Image
from resmip.image.data_types import ImageDTypeLike
from resmip.image.metadata import DicomModality
from resmip.utils import PathLike

from .utils import get_structure_name_from_filename

__all__ = ["RTStructure"]

logger = logging.getLogger(__name__)


class RTStructure(Image):
    """RT Structure (wrapper of resmip.Image)."""

    def __init__(self, *args, name: str = "", **kwargs):
        """Call resmip.Image constructor and set a name for the RT Structure.

        Args:
            *args: arguments provided to ``resmip.Image.__init__``.
            name (str): name of the RT Structure. Defaults to an empty string.
            **kwargs: extra arguments used in ``resmip.Image.__init__``.
        """
        warnings.warn(
            "'RTStructure' is deprecated and will be removed in a future release. "
            "Use 'resmip.Segmentation' instead.",
            FutureWarning,
        )
        if not isinstance(name, str):
            raise ValueError(
                f"type({name}) ({type(name)}) is not a valid type for name. Supported type(s): str."
            )
        super().__init__(*args, modality=DicomModality.rtstruct, **kwargs)
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

    @property
    def modality(self):
        """DICOM modality of the RT Structure.

        Only "RTSTRUCT" is supported.
        """
        image_modality = self._get_modality()
        if image_modality != DicomModality.rtstruct.value:
            raise ValueError(f"{image_modality} is not a valid modality value for RT structures.")
        return image_modality

    def astype(self, dtype: ImageDTypeLike) -> RTStructure:
        """Convert pixel array type to the specified value, by casting a new structure.

        Args:
            dtype (ImageDTypeLike): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global ``FORMAT_TO_TYPESTR`` dictionary.

        Returns:
            RTStructure: new structure with specified data type.
        """
        return RTStructure(super().astype(dtype=dtype), name=self.name)

    @classmethod
    def read(
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
        new_rt_structure = cls(Image.read(filename), name=structure_name)
        return new_rt_structure

    @classmethod
    def read_image(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        structure_name: str | None = None,
        reference_image: Image | None = None,
    ) -> RTStructure:  # pragma: no cover
        """Read RT Structure from file.

        The image format is automatically determined from filename's suffix.

        .. warning::
            ``RTStructure.read_image`` is deprecated and will be removed in a future release.
            Use ``RTStructure.read`` instead.

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
        warnings.warn(
            "'RTStructure.read_image' is deprecated and will be removed in a future release. "
            "Use 'RTStructure.read' instead.",
            FutureWarning,
        )
        return cls.read(
            filename=filename,
            read_metadata=read_metadata,
            structure_name=structure_name,
            reference_image=reference_image,
        )

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
            **kwargs: extra arguments used in ``resmip.Image.__init__``.

        Returns:
            RTStructure: New structure.
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

    def write(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = False,
        use_existing_ids: bool = False,
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
            use_existing_ids (bool): If true, generate unique uids. Currently not used.
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
            return self._write_nondicom(filename, file_format=file_format)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return RTStructureSet([self]).write(
                filename,
                file_format=file_format,
                reference_image_path=reference_image_path,
                series_description=series_description,
            )

    def write_image(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = False,
        file_format: str | None = None,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:  # pragma: no cover
        """Save RT Structure file.

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        .. warning::
            ``RTStructure.write_image`` is deprecated and will be removed in a future release.
            Use ``RTStructure.write`` instead.

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
        warnings.warn(
            "'RTStructure.write_image' is deprecated and will be removed in a future release. "
            "Use 'RTStructure.write' instead.",
            FutureWarning,
        )
        return self.write(
            filename=filename,
            write_metadata=write_metadata,
            file_format=file_format,
            reference_image_path=reference_image_path,
            series_description=series_description,
        )

    def _write_nondicom(
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

    def resample_onto(
        self,
        reference_image: Image,
        transform: sitk.Transform | None = None,
        *,
        interpolator: int = sitk.sitkNearestNeighbor,
        default_pixel_value: float = 0.0,
    ) -> RTStructure:
        """Resample the structure onto another image's grid by applying a transform.

        Behaves like ``Image.resample_onto`` -- the returned structure adopts the
        grid (size, spacing, origin, direction) of ``reference_image``, and the
        transform follows the SimpleITK reverse-mapping convention
        (``output(p) = self(transform(p))``); when ``transform`` is ``None`` an
        identity transform is used, reducing the operation to pure grid
        resampling. This is the operation used to propagate a structure onto a
        reference frame, e.g. carrying it through a registration result.

        Nearest-neighbour interpolation is used by default to preserve the
        discrete label values of the mask. Passing any other interpolator emits a
        warning, as interpolation may introduce voxel values absent from the input.

        Args:
            reference_image (Image): Image whose grid defines the output sampling
                geometry.
            transform (sitk.Transform | None): Transform mapping reference-space
                points into self-space. If ``None``, an identity transform is used.
            interpolator (int): SimpleITK interpolator constant. Defaults to
                ``sitk.sitkNearestNeighbor``; other values are discouraged for
                RT structures and trigger a warning.
            default_pixel_value (float): Value assigned to output voxels whose
                sampling coordinate falls outside the extent of ``self``.

        Returns:
            RTStructure: New structure sampled onto ``reference_image``'s grid,
                preserving this structure's name.
        """
        if interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "Only sitk.sitkNearestNeighbor should be used when resampling RT structures."
            )
        resampled_image = super().resample_onto(
            reference_image=reference_image,
            transform=transform,
            interpolator=interpolator,
            default_pixel_value=default_pixel_value,
        )
        return RTStructure(resampled_image, name=self.name)

    def reorient(
        self,
        new_direction: np.ndarray | tuple[float, ...] | None = None,
        *,
        interpolator: int = sitk.sitkNearestNeighbor,
        default_pixel_value: float = 0,
        allow_reflection: bool = False,
        flip_axis: int = 0,
    ) -> RTStructure:
        """Rotate the structure to match a target direction cosine matrix.

        Computes the rotation matrix that maps the current image direction
        to ``new_direction`` and applies it via ``rotate``. Only pure
        rotations (det = +1) are supported; improper rotations (det = -1,
        e.g. reflections) are optionally supported. Transformations with
        -1 < det < 1 raise ``NotImplementedError``.

        Args:
            new_direction (np.ndarray | tuple[float, ...] | None): Target
                direction as a 9-element flattened row-major rotation matrix.
                If ``None``, defaults to the identity
                direction (standard axial orientation).
            interpolator (int): SimpleITK interpolator constant forwarded
                to ``rotate``.
                Only nearest neighbors should be used for RT structures.
            default_pixel_value (float): Fill value for voxels outside the
                original image extent, forwarded to ``rotate``.
            allow_reflection (bool): Allow reflections for rotations with
                negative determinant. If set to false, improper rotations
                raise an exception.
            flip_axis (int): Internal reflection axis for improper rotations.

        Returns:
            RTStructure: Reoriented structure matching ``new_direction``.
        """
        return RTStructure(
            super().reorient(
                new_direction=new_direction,
                interpolator=interpolator,
                default_pixel_value=default_pixel_value,
                allow_reflection=allow_reflection,
                flip_axis=flip_axis,
            ),
            name=self.name,
        )

    def _flip(self, axis: int) -> RTStructure:
        """Reverse the voxel order along one image axis, mirroring the image.

        The voxels are reversed along the given axis while the direction and
        origin are updated so the image occupies the same physical extent as
        before; only its handedness changes (the determinant of the direction
        cosine matrix flips sign). The operation is lossless and is its own
        inverse: flipping twice along the same axis restores the original image.

        Args:
            axis (int): Image axis to reverse, in (x, y, z) order
                (``x=0``, ``y=1``, ``z=2``).

        Returns:
            RTStructure: Mirrored structure with the same spacing, size and metadata, and
                a direction cosine matrix of opposite determinant.
        """
        return RTStructure(super()._flip(axis=axis), name=self.name)

    def rotate(
        self,
        angle_x: float = 0,
        angle_y: float = 0,
        angle_z: float = 0,
        *,
        interpolator: int = sitk.sitkNearestNeighbor,
        default_pixel_value: float = 0,
    ) -> RTStructure:
        """Apply a 3D rotation to the structure and return the transformed copy.

        Rotations are applied in the intrinsic order Y -> X -> Z, centred on
        the middle voxel of the image in physical coordinates. The output
        image has the same size and spacing as the input. Direction and
        origin are updated to reflect the rotation.

        Args:
            angle_x (float): Rotation angle around the X axis in radians.
            angle_y (float): Rotation angle around the Y axis in radians.
            angle_z (float): Rotation angle around the Z axis in radians.
            interpolator (int): SimpleITK interpolator constant used during
                resampling.
                Only nearest neighbors should be used for RT structures.
            default_pixel_value (float): Value used for voxels outside the
                original image extent after rotation.

        Returns:
            RTStructure: Rotated structure with updated direction and origin, and the
                same size, spacing, and metadata as the original.
        """
        if interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "Only sitk.sitkNearestNeighbor should be used when resampling RT structures."
            )
        return RTStructure(
            super().rotate(
                angle_x=angle_x,
                angle_y=angle_y,
                angle_z=angle_z,
                interpolator=interpolator,
                default_pixel_value=default_pixel_value,
            ),
            name=self.name,
        )

    def pad(self, reference_image: Image, **kwargs) -> RTStructure:
        """Pad the structure on top of another image.

        Uses the same notation as ``numpy.pad``.
        The struct is shifted aligning its top-left voxel with the reference image.
        The two images must have the same voxel spacing.
        The shifted structure is cropped if it extends out of the reference image.

        Args:
            reference_image (Image): Image used as reference for padding.
            **kwargs: same arguments used in ``np.pad``.

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
        warnings.warn(
            "'RTStructureSet' is deprecated and will be removed in a future release. "
            "Use 'resmip.RTStructureSet' from 'resmip.segmentation' instead.",
            FutureWarning,
        )
        if structures is None:
            structures = []
        self.update({structure.name: structure for structure in structures})

    @classmethod
    def read(
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
            regex (bool): Whether to consider ``structure_names`` as a regular expression or not.
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
            structures.append(RTStructure.read(f))
        return cls(structures)

    @classmethod
    def read_image(
        cls,
        filename: PathLike | list[PathLike],
        *,
        structure_names: list[str] | None = None,
        regex: bool = False,
        reference_image: Image | None = None,
        parallel: bool = True,
    ) -> RTStructureSet:  # pragma: no cover
        """Read RT Structure Set file(s).

        .. warning::
            ``RTStructureSet.read_image`` is deprecated and will be removed in a future release.
            Use ``RTStructureSet.read`` instead.

        Args:
            filename (PathLike | list[PathLike]): Name of the DICOM RT structure set.
                If reading from NIfTI, use a list of paths to the structures,
            structure_names (list[str] | None): Names of the structures to be read.
                Used for reading only specific structures in a dicom files,
                can also be a regular expression.
            regex (bool): Whether to consider ``structure_names`` as a regular expression or not.
            parallel (bool): Whether to read structures in parallel or not.
            reference_image (Image | None): 3D image used as reference for dicom Structures
                (not used for other formats).

        Returns:
            RTStructureSet: RT Structure Set.
        """
        warnings.warn(
            "'RTStructureSet.read_image' is deprecated and will be removed in a future release. "
            "Use 'RTStructureSet.read' instead.",
            FutureWarning,
        )
        return cls.read(
            filename=filename,
            structure_names=structure_names,
            regex=regex,
            reference_image=reference_image,
            parallel=parallel,
        )

    def write(
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
            if not filename.is_dir():
                dicom_rtst.write(
                    self, filename, reference_image_path, series_description=series_description
                )
                return
            if file_format is None:
                raise ValueError(
                    "File format must be specified when "
                    "saving a non-DICOM RT structure set to a directory."
                )
            filename = [filename / f"{struct_name}{file_format}" for struct_name in self]
        filename = [Path(f) for f in filename]
        if len(filename) != len(self):
            raise ValueError(
                "The number of filenames provided is different than the number of structures."
            )
        for structure_filename, structure in zip(filename, self.values()):
            structure.write(structure_filename, file_format=file_format)

    def write_image(
        self,
        filename: PathLike | list[PathLike],
        *,
        file_format: str | None = None,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:  # pragma: no cover
        """Save RT Structure Set file(s).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        .. warning::
            ``RTStructureSet.write_image`` is deprecated and will be removed in a future release.
            Use ``RTStructureSet.write`` instead.

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
        warnings.warn(
            "'RTStructureSet.write_image' is deprecated and will be removed in a future release. "
            "Use 'RTStructureSet.write' instead.",
            FutureWarning,
        )
        return self.write(
            filename=filename,
            file_format=file_format,
            reference_image_path=reference_image_path,
            series_description=series_description,
        )
