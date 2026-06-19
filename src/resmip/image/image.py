"""Image object containing pixel array and metadata."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from pydicom.uid import generate_uid

import resmip.dicom_utils.series as dicom_series
from resmip.dicom_utils import string_tag_for_keyword
from resmip.image.coregistration import CoregistrationMetric
from resmip.image.data_types import (
    ImageDTypeLike,
    datatype_from_id,
    is_unsigned,
    sitk_image_dtype,
)
from resmip.image.dicom_fields import PATIENT_RELATED_FIELDS, STUDY_RELATED_FIELDS
from resmip.image.metadata import (
    SERIES_MODALITIES,
    DicomModality,
)
from resmip.utils import PathLike, format_digit_string

from .transforms import _rx, _ry, _rz

__all__ = ["Image"]

logger = logging.getLogger(__name__)


def _metadata_file_name(filename: PathLike) -> Path:
    """Generate the filename for the metadata.

    Defaults a json file with same name of the output image file (filename).
    The json filename is prepended with a "." to make it hidden.

    Args:
        filename (PathLike): name of the output image file name.

    Returns:
        Path: Path of the json metadata file.
    """
    filename = Path(filename)
    return filename.parent / f".{filename.stem}.json"


def _validate_modality(image_modality: str | DicomModality) -> DicomModality:
    """Check if the image modality saved in metadata is valid."""
    if image_modality == "":
        logger.warning("Image modality must be defined.")
        return DicomModality.ct

    if isinstance(image_modality, str):
        image_modality = getattr(DicomModality, image_modality.lower())

    if image_modality not in SERIES_MODALITIES:
        raise ValueError(
            f"The provided modality ({image_modality}) is not a valid DICOM series modality."
        )
    return image_modality


class Image(sitk.Image):
    """Wrapper class of SimpleITK.Image with support to headers."""

    def __init__(
        self,
        *args,
        metadata: dict[str, str] | None = None,
        modality: DicomModality | str | None = None,
    ):
        """Call sitk.Image constructor and create an empty dictionary for the header."""
        super().__init__(*args)
        self._metadata = self._generate_minimal_empty_metadata()
        """Dictionary containing metadata."""
        # Copy metadata when creating an image from an existing one
        if len(args) > 0:
            if isinstance(args[0], Image):
                self._metadata = args[0].metadata
        if metadata:
            self._metadata.update(metadata)
        if modality:
            if isinstance(modality, DicomModality):
                modality = modality.value
            self._metadata[string_tag_for_keyword("Modality")] = modality

    def _generate_minimal_empty_metadata(self) -> dict[str, str]:
        """Initialize metadata with required fields."""
        return {
            string_tag_for_keyword("Modality"): "",
            string_tag_for_keyword("PatientID"): "",
            string_tag_for_keyword("StudyInstanceUID"): "",
            string_tag_for_keyword("SeriesInstanceUID"): "",
        }

    def generate_ids(self, force: bool = False):
        """Generate study/series ids.

        Args:
            force (bool): If true, generate missing values and
                force-generate a new SeriesInstanceUID.
                If false, re-use current identifiers if set,
                otherwise generate new values.
        """
        if self.study_instance_uid == "":
            self._metadata[string_tag_for_keyword("StudyInstanceUID")] = generate_uid()
        if self.series_instance_uid == "" or force:
            self._metadata[string_tag_for_keyword("SeriesInstanceUID")] = generate_uid()

    def _optionally_transfer_information(self, other: Image, fields_to_copy: list[str]):
        """Copy information from the other image.

        Only fields that are present in the other image are copied.

        Args:
            other (Image): Other image for association.
            fields_to_copy (list[str]): List of DICOM fields to optionally copy.
        """
        for dicom_field in fields_to_copy:
            field_tag = string_tag_for_keyword(dicom_field)
            if field_tag in other.metadata:
                self._metadata[field_tag] = other.metadata[field_tag]

    def associate_to(self, other: Image, *, level: str = "study"):
        """Associate the current image to another image.

        Copy identifiers.

        Args:
            other (Image): The other image from which
                to copy information.
            level (str): One of the following:
                - "patient": copy only patient-related information
                - "study": copy patient- and study-related information
        """
        if level not in ["patient", "study"]:
            raise ValueError(
                f"{level} is not a supported level. Supported values are 'patient', 'study'."
            )
        # force-copy patient id as it is strictly required
        self._metadata[string_tag_for_keyword("PatientID")] = other.patient_id
        self._optionally_transfer_information(other, fields_to_copy=PATIENT_RELATED_FIELDS)
        if level == "study":
            self._metadata[string_tag_for_keyword("StudyInstanceUID")] = other.study_instance_uid
            self._optionally_transfer_information(other, fields_to_copy=STUDY_RELATED_FIELDS)

    def __getitem__(self, key) -> Image:
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
        return Image(super().__getitem__(key), metadata=self.metadata)

    @property
    def metadata(self) -> dict[str, str]:
        """Dicom header combined with other metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in mm (x, y, z)."""
        return self.GetSpacing()

    @spacing.setter
    def spacing(self, value: tuple[float, float, float]):
        self.SetSpacing(value)

    @property
    def origin(self) -> tuple[float, float, float]:
        """Coordinates of the top left voxel in mm (x, y, z)."""
        return self.GetOrigin()

    @origin.setter
    def origin(self, value: tuple[float, float, float]):
        """The original DICOM header key is not updated."""
        self.SetOrigin(value)

    @property
    def direction(self) -> tuple[float]:
        """Direction cosine matrix.

        For more information, see here:
        https://dicom.innolitics.com/ciods/rt-dose/image-plane/00200037
        """
        return self.GetDirection()

    @direction.setter
    def direction(self, value: tuple[float]):
        self.SetDirection(value)

    @property
    def size(self) -> tuple[int, int, int]:
        """Image size in pixels."""
        return self.GetSize()

    def _get_modality(self) -> str:
        """Obtain DICOM modality from image metadata."""
        return self._metadata[(string_tag_for_keyword("Modality"))]

    @property
    def modality(self) -> str:
        """Image modality."""
        image_modality = self._get_modality()
        return _validate_modality(image_modality).value

    @property
    def patient_id(self) -> str:
        """Patient ID.

        Defaults to an empty string if not set.
        """
        return self._metadata[(string_tag_for_keyword("PatientID"))].strip(" ")

    @property
    def study_instance_uid(self) -> str:
        """Study Instance UID.

        Defaults to an empty string if not set.
        """
        return self._metadata[(string_tag_for_keyword("StudyInstanceUID"))]

    @property
    def series_instance_uid(self) -> str:
        """Series Instance UID.

        Defaults to an empty string if not set.
        """
        return self._metadata[(string_tag_for_keyword("SeriesInstanceUID"))]

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        direction: tuple[float],
        metadata: dict[str, str] | None = None,
        **kwargs,
    ) -> Image:
        """Create a new image from a numpy array.

        Args:
            array (np.ndarray): 3D array containing voxel values for the image (z, y, x).
            spacing (tuple[float, float, float]): Voxel spacing for the image in mm (x, y, z).
            origin (tuple[float, float, float]): Coordinates of the top left voxel in mm (x, y, z).
            direction (tuple[float]): Direction cosine matrix.
            metadata (dict[str, str] | None): Metadata containing information from the DICOM header.
            **kwargs: extra arguments used in ``resmip.Image.__init__``.

        Returns:
            Image: New image
        """
        new_image = cls(sitk.GetImageFromArray(array), **kwargs)
        if metadata is not None:
            new_image.metadata = metadata
        new_image.spacing = spacing
        new_image.origin = origin
        new_image.direction = direction
        return new_image

    def __array__(
        self,
        dtype: str | npt.DTypeLike | None = None,
        copy: bool | None = None,
        view: bool | None = None,
    ) -> np.ndarray:
        """Convert an image to a numpy array.

        Wrapper of sitk.GetArrayFromImage().

        .. note::
            ``copy`` and ``view`` are mutually exclusive.
        .. warning::
            ``view`` is deprecated and will be removed in a future release.

        Args:
            dtype (str | npt.DTypeLike | None): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global ``FORMAT_TO_TYPESTR`` dictionary.
            copy (bool | None): If True, then new array is copied to memory.
                If None then the array is copied only if needed, i.e. if type casting
                specified by ``dtype`` is required.
                For False it raises a ValueError if a copy cannot be avoided.
            view (bool | None): If set to true, return a view of the underlying data,
                without copying them. If a dtype is specified, a copy is returned anyway.

        Returns:
            np.ndarray: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        """
        if copy is not None and view is not None:
            raise ValueError(
                "__array__() received both 'copy' and deprecated 'view'. Use only 'copy'."
            )
        if view is not None:
            if view is True:
                warning_message = "Use 'copy=False' instead."
            else:
                warning_message = "Use 'copy=True' instead."
            warnings.warn(
                "'view' is deprecated and will be removed in a future release. " + warning_message
            )
            copy = not view

        needs_copy = False
        if dtype:
            if dtype != self.dtype:
                needs_copy = True

        if copy is False:
            if needs_copy:
                raise ValueError(
                    "Cannot return a view when asking to cast "
                    f"the image to a different dtype ({dtype}). "
                    f"Current dtype is {self.dtype}."
                )
            return sitk.GetArrayViewFromImage(self)
        if copy is True or needs_copy:
            image_array = sitk.GetArrayFromImage(self)
            if dtype is not None:
                image_array = image_array.astype(dtype=dtype)
            return image_array
        return sitk.GetArrayViewFromImage(self)

    def numpy(
        self,
        dtype: str | npt.DTypeLike | None = None,
        copy: bool | None = True,
        view: bool | None = None,
    ) -> np.ndarray:
        """Generate a numpy array of pixels from the image.

        Wrapper of sitk.GetArrayFromImage().

        .. note::
            ``copy`` and ``view`` are mutually exclusive.
        .. warning::
            ``view`` is deprecated and will be removed in a future release.

        Args:
            dtype (str | npt.DTypeLike | None): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global ``FORMAT_TO_TYPESTR`` dictionary.
            copy (bool | None): If True, then new array is copied to memory.
                If None then the array is copied only if needed, i.e. if type casting
                specified by ``dtype`` is required.
                For False it raises a ValueError if a copy cannot be avoided.
            view (bool | None): If set to true, return a view of the underlying data,
                without copying them. If a dtype is specified, a copy is returned anyway.

        Returns:
            np.ndarray: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        """
        if view is not None:
            if view is True:
                warning_message = "Use 'copy=False' instead."
            else:
                warning_message = "Use 'copy=True' instead."
            warnings.warn(
                "'view' is deprecated and will be removed in a future release. " + warning_message
            )
            copy = not view
            view = None
        return self.__array__(dtype=dtype, copy=copy, view=view)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Data type of the Image."""
        return datatype_from_id(self.GetPixelID())

    def astype(self, dtype: ImageDTypeLike) -> Image:
        """Convert pixel array type to the specified value, by casting a new image.

        Args:
            dtype (ImageDTypeLike): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global ``FORMAT_TO_TYPESTR`` dictionary.

        Returns:
            Image: new image with specified data type.
        """
        return Image(sitk.Cast(self, sitk_image_dtype(dtype)), metadata=self.metadata)

    @classmethod
    def read(cls, filename: PathLike, read_metadata: bool = True) -> Image:
        """Read an image from a file or directory.

        The image format is automatically determined from the type of path
        provided. DICOM series are read from directories; all other formats
        are read from single files.

        Args:
            filename (PathLike): Path to the image file or directory.
                If a directory is provided, it is read as a DICOM series
                and metadata is extracted from the DICOM headers.
                Otherwise, the format is inferred from the file suffix
                (e.g. ``.nii.gz``, ``.mha``) and metadata is read from
                a sidecar JSON file named ``.{stem}.json`` if present.
            read_metadata (bool): If ``True``, read metadata from the
                sidecar JSON file. Has no effect for DICOM series, where
                metadata is always read from the DICOM headers.

        Returns:
            Image: Image with metadata populated from the DICOM headers
                or sidecar JSON file.
        """
        filename = Path(filename)
        if filename.is_dir():
            sitk_image, series_metadata = dicom_series.read(filename)
            return cls(sitk_image, metadata=series_metadata)
        return Image._read_nondicom(filename=filename, read_metadata=read_metadata)

    @classmethod
    def _read_nondicom(cls, filename: PathLike, read_metadata: bool = True) -> Image:
        """Read an image from a non-DICOM file.

        Reads the image using SimpleITK, inferring the format from the
        file suffix. If a sidecar JSON metadata file named ``.{stem}.json``
        exists alongside the image file and ``read_metadata`` is ``True``,
        its contents are merged into the metadata dictionary. Tags already
        present in the SimpleITK image header are always included and will
        override sidecar values for duplicate keys. If no ``Modality`` tag
        is found, it is set to an empty string.

        Args:
            filename (PathLike): Path to the image file. The format is
                inferred from the file suffix (e.g. ``.nii.gz``, ``.mha``).
            read_metadata (bool): If ``True``, read metadata from the
                sidecar JSON file named ``.{stem}.json`` if it exists.

        Returns:
            Image: Image with metadata populated from the image header
                and sidecar JSON file.
        """
        sitk_image = sitk.ReadImage(filename)
        if read_metadata and _metadata_file_name(filename).exists():
            serialized_metadata = _metadata_file_name(filename).read_text()
            series_metadata = json.loads(serialized_metadata)
        else:
            series_metadata = {}
        for key in sitk_image.GetMetaDataKeys():
            value = sitk_image.GetMetaData(key)
            value = format_digit_string(value)
            series_metadata[key] = value
        if string_tag_for_keyword("Modality") not in series_metadata:
            series_metadata[string_tag_for_keyword("Modality")] = ""
        return cls(sitk_image, metadata=series_metadata)

    @classmethod
    def read_image(
        cls, filename: PathLike, read_metadata: bool = True
    ) -> Image:  # pragma: no cover
        """Load image file (and metadata).

        The image format is automatically determined from filename's suffix.

        .. warning::
            ``Image.read_image`` is deprecated and will be removed in a future release.
            Use ``Image.read`` instead.

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                the reader assumes to read a Dicom series. Otherwise, it assumes a metatadata
                file with the following format exists: f".{filename.stem}.json".
            read_metadata (bool): If true, read the json file with metadata
                (not applicable for dicom files).

        Returns:
            Image: Image and metadata.
        """
        warnings.warn(
            "'Image.read_image' is deprecated and will be removed in a future release. "
            "Use 'Image.read' instead."
        )
        return cls.read(filename=filename, read_metadata=read_metadata)

    def write(
        self, filename: PathLike, *, write_metadata: bool = True, use_existing_ids: bool = False
    ) -> None:
        """Save image file (and metadata).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                the writer assumes to write a Dicom series.
            write_metadata (bool): If true, save the json file with metadata
                (not applicable for dicom files).
            use_existing_ids (bool): If true, re-use current identifiers if set,
                otherwise generate new values.
                If false, generate missing values and force-generate a new
                SeriesInstanceUID.
        """
        filename = Path(filename)
        if not filename.exists() and filename.suffix == "":
            filename.mkdir(parents=True, exist_ok=True)
        if filename.is_dir():
            self.generate_ids(force=not use_existing_ids)
            dicom_series.write(self, self.metadata, filename)
            return
        filename.parent.mkdir(parents=True, exist_ok=True)
        self._write_nondicom(filename=filename, write_metadata=write_metadata)

    def write_image(
        self, filename: PathLike, *, write_metadata: bool = True
    ) -> None:  # pragma: no cover
        """Save image file (and metadata).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        .. warning::
            ``Image.write_image`` is deprecated and will be removed in a future release.
            Use ``Image.write`` instead.

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                the writer assumes to write a Dicom series.
            write_metadata (bool): If true, save the json file with metadata
                (not applicable for dicom files).
        """
        warnings.warn(
            "'Image.write_image' is deprecated and will be removed in a future release. "
            "Use 'Image.write' instead."
        )
        return self.write(filename=filename, write_metadata=write_metadata)

    def _write_nondicom(self, filename: PathLike, write_metadata: bool = True) -> None:
        """Save image file (and metadata) to non-DICOM formats using ITK.

        Args:
            filename (PathLike): Name of the file. If filename is a directory,
                the writer assumes to write a Dicom series.
            write_metadata (bool): If true, save the json file with metadata
                (not applicable for dicom files).
        """
        sitk.WriteImage(self, filename)
        if write_metadata:
            serialized_metadata = json.dumps(self.metadata)
            _metadata_file_name(filename).write_text(serialized_metadata)

    def resample(
        self,
        new_spacing: Iterable,
        interpolator: int = sitk.sitkLinear,
        default_pixel_value: float = 0,
    ) -> Image:
        """Resample the image with a new voxel spacing (in mm).

        Args:
            new_spacing (Iterable): New voxel spacing of the resampled image (x, y, z) in mm.
            interpolator (int): Interpolation method used for image resampling.
            default_pixel_value (float): Default value for pixel intensity.

        Returns:
            Image: Resampled image.
        """
        if isinstance(new_spacing, (list, tuple)):
            new_spacing = np.array(new_spacing)
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(default_pixel_value)
        resampler.SetOutputDirection(self.direction)
        resampler.SetOutputOrigin(self.origin)
        resampler.SetOutputSpacing(new_spacing.tolist())

        orig_size = np.array(self.GetSize(), dtype=int)
        orig_spacing = self.spacing
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(int)  # Image dimensions are in integers
        new_size = [int(s) for s in new_size]
        resampler.SetSize(new_size)

        new_img = Image(resampler.Execute(self))
        new_img.metadata = self.metadata
        new_img.metadata[string_tag_for_keyword("PixelSpacing")] = "\\".join(
            [str(x) for x in new_img.spacing[:2]]
        )
        new_img.metadata[string_tag_for_keyword("SliceThickness")] = str(new_img.spacing[2])
        return new_img

    @staticmethod
    def _cosine_matrix_from_direction(direction: np.ndarray | tuple[float, ...]) -> np.ndarray:
        """Reshape a flat direction array into a 3x3 direction cosine matrix.

        Args:
            direction (np.ndarray | tuple[float, ...]): Flattened 9-element
                direction array in row-major order, as returned by
                ``sitk.Image.GetDirection()``.

        Raises:
            ValueError: If ``direction`` is not one-dimensional.
            NotImplementedError: If ``direction`` does not have exactly 9
                elements (i.e. the image is not 3D).

        Returns:
            np.ndarray: 3x3 direction cosine matrix where each row is a
                direction cosine vector.
        """
        if not isinstance(direction, np.ndarray):
            direction = np.array(direction)
        if direction.ndim != 1:
            raise ValueError("Only one dimensional direction arrays are supported.")
        if direction.shape != (9,):
            raise NotImplementedError("Only 3D images are currently supported.")
        return direction.reshape(3, 3).copy()

    @property
    def _cosine_matrix(self) -> np.ndarray:
        """3x3 direction cosine matrix of the image."""
        return Image._cosine_matrix_from_direction(self.direction)

    def reorient(
        self,
        new_direction=None,
        interpolator: int = sitk.sitkBSpline,
        default_pixel_value: float = 0,
    ) -> Image:
        """Rotate the image to match a target direction cosine matrix.

        Computes the rotation matrix that maps the current image direction
        to ``new_direction`` and applies it via ``rotate``. Only pure
        rotations (det = +1) are supported; improper rotations (det = -1,
        e.g. reflections) raise ``NotImplementedError``.

        Args:
            new_direction (np.ndarray | tuple[float, ...] | None): Target
                direction as a 9-element flattened row-major rotation matrix.
                If ``None``, defaults to the identity
                direction (standard axial orientation).
            interpolator (int): SimpleITK interpolator constant forwarded
                to ``rotate``.
            default_pixel_value (float): Fill value for voxels outside the
                original image extent, forwarded to ``rotate``.

        Raises:
            NotImplementedError: If the computed rotation is an improper
                rotation (determinant = -1) or is not a valid rotation matrix
                (determinant ≠ ±1).

        Returns:
            Image: Reoriented image matching ``new_direction``.
        """
        if new_direction is None:
            new_direction = np.eye(3).flatten()

        # generate overall transformation
        current_direction = self._cosine_matrix
        final_direction = Image._cosine_matrix_from_direction(new_direction)
        overall_rotation = final_direction @ np.linalg.inv(current_direction)

        # check if the overall rotation is a pure rotation:
        # det(R) = +1
        if np.isclose(np.linalg.det(overall_rotation), -1):
            # flip the rotation
            raise NotImplementedError("Currently only pure rotations are supported.")

        if not np.isclose(np.linalg.det(overall_rotation), 1):
            raise NotImplementedError("Currently only pure rotations are supported.")

        angle_x = np.arctan2(
            overall_rotation[2, 1],
            np.sqrt(overall_rotation[2, 0] ** 2 + overall_rotation[2, 2] ** 2),
        )
        angle_y = np.arctan2(-overall_rotation[2, 0], overall_rotation[2, 2])
        angle_z = np.arctan2(-overall_rotation[0, 1], overall_rotation[1, 1])
        return self.rotate(
            angle_x=angle_x,
            angle_y=angle_y,
            angle_z=angle_z,
            interpolator=interpolator,
            default_pixel_value=default_pixel_value,
        )

    def _flip(self, axis: int) -> Image:
        """Reverse voxel order along an image axis (x=0, y=1, z=2). Lossless reindex.

        Flips the data and adjusts direction/origin so the image is physically
        unchanged but its direction determinant flips sign. Its own inverse.
        """
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
            Image: Mirrored image with the same spacing, size and metadata, and
                a direction cosine matrix of opposite determinant.

        """
        if axis not in (0, 1, 2):
            raise ValueError(f"flip_axis must be 0, 1 or 2; got {axis}.")
        data = np.flip(self.numpy(), axis=2 - axis).copy()  # image axis -> numpy axis
        direction = self._cosine_matrix
        spacing = np.array(self.spacing)
        size = np.array(self.size)
        new_origin = np.array(self.origin) + direction[:, axis] * spacing[axis] * (size[axis] - 1)
        direction[:, axis] *= -1
        return Image.from_array(
            data,
            spacing=self.spacing,
            origin=tuple(new_origin),
            direction=tuple(direction.flatten()),
            metadata=self.metadata,
        )

    def rotate(
        self,
        angle_x: float = 0,
        angle_y: float = 0,
        angle_z: float = 0,
        *,
        interpolator: int = sitk.sitkBSpline,
        default_pixel_value: float = 0,
    ) -> Image:
        """Apply a 3D rotation to the image and return the transformed copy.

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
            default_pixel_value (float): Value used for voxels outside the
                original image extent after rotation.

        Returns:
            Image: Rotated image with updated direction and origin, and the
                same size, spacing, and metadata as the original.
        """
        transform = sitk.Euler3DTransform()
        rotation_center = np.array(
            self.TransformContinuousIndexToPhysicalPoint([(sz - 1) / 2 for sz in self.size])
        )
        transform.SetCenter(rotation_center)
        transform.SetRotation(angleX=angle_x, angleY=angle_y, angleZ=angle_z)
        new_img = Image(
            sitk.Resample(
                self, self, transform, interpolator, default_pixel_value, self.GetPixelID()
            ),
            metadata=self.metadata,
            modality=self.modality,
        )
        rotation_matrix = _rz(angle_z) @ _rx(angle_x) @ _ry(angle_y)
        new_img.direction = (rotation_matrix @ self._cosine_matrix).flatten()
        new_img.origin = transform.TransformPoint(self.origin)
        return new_img

    def pad(self, reference_image: Image, **kwargs) -> Image:
        """Pad the image on top of another image.

        Uses the same notation as ``numpy.pad``.
        The image is shifted aligning its top-left voxel with the reference image.
        The two images must have the same voxel spacing.
        The shifted image is cropped if it extends out of the reference image.

        Args:
            reference_image (Image): Image used as reference for padding.
            **kwargs: same arguments used in ``np.pad``.

        Returns:
            Image: New image with same shape and spacing of the reference.
        """
        if self.spacing != reference_image.spacing:
            raise ValueError(
                f"Both images must have the same voxel spacing. The image has {self.spacing},"
                f"the reference image has {reference_image.spacing}."
            )
        if np.any(np.round(self.direction, 4) != np.round(reference_image.direction, 4)):
            raise ValueError(
                f"Both images must have the same direction. The image has {self.direction},"
                f"the reference image has {reference_image.direction}."
            )
        xyz_lower_padding = np.array(
            [
                int(x)
                for x in (np.array(self.origin) - np.array(reference_image.origin))
                / reference_image.spacing
            ]
        )
        xyz_upper_padding = (
            np.array(reference_image.GetSize()) - np.array(self.GetSize()) - xyz_lower_padding
        )

        lower_boundary_crop = []
        for i, padding_value in enumerate(xyz_lower_padding):
            crop_value = 0
            if padding_value < 0:
                crop_value = -padding_value
                xyz_lower_padding[i] = 0
            lower_boundary_crop.append(crop_value)
        upper_boundary_crop = []
        for i, padding_value in enumerate(xyz_upper_padding):
            crop_value = None
            if padding_value < 0:
                crop_value = padding_value
                xyz_upper_padding[i] = 0
            upper_boundary_crop.append(crop_value)
        boundary_crop = []
        for lower_bound, upper_bound in zip(lower_boundary_crop, upper_boundary_crop):
            boundary_crop.append(slice(lower_bound, upper_bound))
        img_arr = self.numpy()[tuple(np.flip(boundary_crop))]

        xyz_padding = np.flip(np.stack([xyz_lower_padding, xyz_upper_padding], axis=1), axis=0)
        new_arr = np.pad(img_arr, xyz_padding, **kwargs)
        return Image.from_array(
            new_arr,
            spacing=reference_image.spacing,
            origin=reference_image.origin,
            direction=reference_image.direction,
            metadata=self.metadata,
        )

    @staticmethod
    def _get_coregistration_method(
        seed: int = 0,
        num_threads: int | None = None,
        metric: CoregistrationMetric = CoregistrationMetric.mutual_information,
    ) -> sitk.ImageRegistrationMethod:
        """Generate method used for coregistration.

        Args:
            seed (int): Random seed for the registration method. When set to 0,
                uses system walltime. Use different values for deterministic behaviour.
            num_threads (int|None): Number of threads used for coregistration. By default, it is
                set to the maximum number of available threads.
                Set it to 1 for deterministic behaviour.
            metric (CoregistrationMetric): metric used for coregistration.

        Returns:
            sitk.ImageRegistrationMethod: Registration method used for coregistration.
        """

        def _set_metric(reg_method: sitk.ImageRegistrationMethod, metric: CoregistrationMetric):
            """Set coregistration metric."""
            number_of_mutual_information_bins = 100
            if metric == CoregistrationMetric.correlation:
                reg_method.SetMetricAsCorrelation()
            elif metric == CoregistrationMetric.mutual_information:
                reg_method.SetMetricAsMattesMutualInformation(
                    numberOfHistogramBins=number_of_mutual_information_bins
                )
            else:
                raise ValueError(f"The provided metric {metric} is not supported.")

        registration_method = sitk.ImageRegistrationMethod()
        if num_threads:
            registration_method.SetGlobalDefaultNumberOfThreads(num_threads)
        _set_metric(registration_method, metric)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01, seed=seed)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-4,
            numberOfIterations=500,
            gradientMagnitudeTolerance=1e-8,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return registration_method

    def coregister(
        self,
        reference_image: Image,
        *,
        fill_value: float = 0.0,
        coregistration_metric: CoregistrationMetric = CoregistrationMetric.mutual_information,
        seed: int = 0,
        num_threads: int | None = None,
    ) -> Image:
        """Coregiser the image on top of another (reference) image.

        Args:
            reference_image (Image): Image used as reference for coregistration.
            fill_value (float): Value used to fill voxels during resampling (defaults to 0).
            coregistration_metric (CoregistrationMetric): metric used for coregistration.
            seed (int): Random seed for the registration method. When set to 0,
                uses system walltime. Use different values for deterministic behaviour.
            num_threads (int|None): Number of threads used for coregistration. By default, it is
                set to the maximum number of available threads.
                Set it to 1 for deterministic behaviour.

        Returns:
            Image: New image coregistered with reference_image.
        """
        registration_method = self._get_coregistration_method(
            seed=seed, num_threads=num_threads, metric=coregistration_metric
        )
        current_type = self.GetPixelID()

        fixed_image = reference_image.astype(np.float32)
        moving_image = self.astype(np.float32)
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        final_transform = registration_method.Execute(fixed_image, moving_image)

        moving_image = Image(
            sitk.Resample(
                moving_image,
                fixed_image,
                final_transform,
                sitk.sitkLinear,
                fill_value,
                moving_image.GetPixelID(),
            ),
            metadata=self.metadata,
        )
        return moving_image.astype(current_type)

    def __add__(self, value: int | float) -> Image:
        """Add constant value to pixel data.

        Args:
            value (int | float): Value to be added to pixel data.

        Returns:
            Image: Image with constant value added to pixel data.
        """
        if np.issubdtype(type(value), np.floating) and not np.issubdtype(self.dtype, np.floating):
            logger.debug("Casting image type to float")
            current_image = self.astype(np.dtype(type(value)))
        else:
            current_image = self
        if value < 0:
            if is_unsigned(current_image.GetPixelID()):
                logger.warning(
                    "Adding a negative value when image "
                    "type is unsigned, make sure to cast it to a signed type "
                    "if pixel values become negative."
                )
        return Image(super(Image, current_image).__add__(value), metadata=self.metadata)

    def __sub__(self, value: int | float) -> Image:
        """Subtract constant value to pixel data.

        Args:
            value (int | float): Value to be subtracted to pixel data.

        Returns:
            Image: Image with constant value subtracted to pixel data.
        """
        if np.issubdtype(type(value), np.floating) and not np.issubdtype(self.dtype, np.floating):
            logger.debug("Casting image type to float")
            current_image = self.astype(np.dtype(type(value)))
        else:
            current_image = self
        if is_unsigned(current_image.GetPixelID()):
            logger.warning(
                "Subtracting value when image "
                "type is unsigned, make sure to cast it to a signed type "
                "if pixel values become negative."
            )
        return Image(super(Image, current_image).__sub__(value), metadata=self.metadata)

    def __mul__(self, value: int | float) -> Image:
        """Multiply constant value to pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            Image: Image with constant value multiplied to pixel data.
        """
        if np.issubdtype(type(value), np.floating) and not np.issubdtype(self.dtype, np.floating):
            logger.debug("Casting image type to float")
            current_image = self.astype(np.dtype(type(value)))
        else:
            current_image = self
        if value < 0:
            if is_unsigned(current_image.GetPixelID()):
                logger.warning(
                    "Multipying a negative value when image "
                    "type is unsigned, make sure to cast it to a signed type "
                    "if pixel values become negative."
                )
        return Image(super(Image, current_image).__mul__(value), metadata=self.metadata)

    def __truediv__(self, value: int | float) -> Image:
        """Divide constant value to pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            Image: Image with constant value multiplied to pixel data.
        """
        logger.debug("Casting image type to float")
        if np.issubdtype(self.dtype, np.integer):
            current_image = self.astype(np.float64)
        else:
            current_image = self
        return Image(super(Image, current_image).__truediv__(value), metadata=self.metadata)
