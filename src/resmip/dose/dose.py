"""RT Dose."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import pydicom.errors
import SimpleITK as sitk

from resmip.image import DicomModality, Image, ImageDTypeLike
from resmip.utils import PathLike

__all__ = ["Dose"]

logger = logging.getLogger(__name__)


class Dose(Image):
    """RT Dose (wrapper of resmip.Image)."""

    def __init__(self, *args, **kwargs):
        """Create new ``Dose`` object."""
        super().__init__(*args, modality=DicomModality.rtdose, **kwargs)

    def __getitem__(self, key) -> Dose:
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
        return Dose(super().__getitem__(key))

    @property
    def modality(self):
        """DICOM modality of the Dose.

        Only "RTDOSE" is supported.
        """
        image_modality = self._get_modality()
        if image_modality != DicomModality.rtdose.value:
            raise ValueError(f"{image_modality} is not a valid modality value for doses.")
        return image_modality

    def astype(self, dtype: ImageDTypeLike) -> Dose:
        """Convert pixel array type to the specified value, by casting a new dose.

        Args:
            dtype (ImageDTypeLike): The dtype to use for the numpy array.
                If None, the default dtype of the image is used
                as defined the global ``FORMAT_TO_TYPESTR`` dictionary.

        Returns:
            Dose: new dose with specified data type.
        """
        return Dose(super().astype(dtype=dtype))

    @classmethod
    def read(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        reference_image: Image | None = None,
    ) -> Dose:
        """Read rt dose from file.

        Doses could have different origin and/or
        spacing compared to the referenced series.
        This function, shifts the dose accordingly.
        Dose values are scaled by "DoseGridScaling", if present.
        https://dicom.innolitics.com/ciods/rt-dose/rt-dose/3004000e

        Args:
            filename (PathLike): Name of the file. If filename ends with ".dcm",
                the reader assumes to read a Dicom rtdose. Otherwise, it assumes a metatadata
                file with the following format exists: f".{filename.stem}.json".
            read_metadata (bool): If true, read the json file with metadata
                (not applicable for dicom files). Currently not used.
            reference_image (Image | None): 3D image used as reference for dicom Doses, in case
                origin and/or spacing differ. When set to None, a warning is raised and no
                shift / resampling is applied.

        Returns:
            Dose: RT Dose.
        """
        image = super().read(filename=filename, read_metadata=read_metadata)
        new_dose = cls(image)
        try:
            dicom_header = pydicom.dcmread(filename)
            try:
                dose_scaling = dicom_header["DoseGridScaling"]
            except KeyError as e:
                raise KeyError(
                    "Missing DoseGridScaling in the DICOM header, "
                    "but it is required in DICOM RT Dose files."
                ) from e
            scaling = float(dose_scaling.value)
        except pydicom.errors.InvalidDicomError:
            logger.info(
                "%s is not a valid DICOM file. Assuming that Dose Scaling is equal to 1.",
                filename,
            )
            scaling = 1
        if reference_image is None:
            logger.warning("No reference image has been provided for the RT Dose.")
            return new_dose
        new_dose = new_dose.resample(new_spacing=reference_image.spacing)
        new_dose = new_dose.pad(reference_image=reference_image)
        return cls(new_dose) * scaling

    @classmethod
    def read_image(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        reference_image: Image | None = None,
    ) -> Dose:  # pragma: no cover
        """Read rt dose from file.

        Doses could have different origin and/or
        spacing compared to the referenced series.
        This function, shifts the dose accordingly.
        Dose values are scaled by "DoseGridScaling", if present.
        https://dicom.innolitics.com/ciods/rt-dose/rt-dose/3004000e

        .. warning::
            ``Dose.read_image`` is deprecated and will be removed in a future release.
            Use ``Dose.read`` instead.

        Args:
            filename (PathLike): Name of the file. If filename ends with ".dcm",
                the reader assumes to read a Dicom rtdose. Otherwise, it assumes a metatadata
                file with the following format exists: f".{filename.stem}.json".
            read_metadata (bool): If true, read the json file with metadata
                (not applicable for dicom files). Currently not used.
            reference_image (Image | None): 3D image used as reference for dicom Doses, in case
                origin and/or spacing differ. When set to None, a warning is raised and no
                shift / resampling is applied.

        Returns:
            Dose: RT Dose.
        """
        warnings.warn(
            "'Dose.read_image' is deprecated and will be removed in a future release. "
            "Use 'Dose.read' instead."
        )
        return cls.read(
            filename=filename, read_metadata=read_metadata, reference_image=reference_image
        )

    def write(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = True,
        use_existing_ids: bool = False,
        file_format: str | None = None,
        # reference_image_path: Optional[PathLike] = None,
    ) -> None:
        """Save RT Dose file.

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        Currently only non-DICOM file formats are supported.

        Args:
            filename (PathLike): Name of the file to be saved.
            write_metadata (bool): If true, write the json file with metadata
                (not applicable for dicom files). Currently not used.
            use_existing_ids (bool): If true, generate unique uids. Currently not used.
            file_format (str | None): Format of the rt dose saved. If None,
                infer it from filename.
            reference_image_path (PathLike | None): Path of the reference dicom image.
                Ignored when saving in formats other than dicom.
        """
        filename = Path(filename)
        if file_format is None:
            file_format = filename.suffix
        filename.parent.mkdir(parents=True, exist_ok=True)
        if file_format != ".dcm":
            return self._write_nondicom(filename, write_metadata=write_metadata)
        raise NotImplementedError("Saving to DICOM RT Dose is currently not supported.")

    def write_image(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = True,
        file_format: str | None = None,
        # reference_image_path: Optional[PathLike] = None,
    ) -> None:  # pragma: no cover
        """Save RT Dose file.

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        Currently only non-DICOM file formats are supported.

        .. warning::
            ``Dose.write_image`` is deprecated and will be removed in a future release.
            Use ``Dose.write`` instead.

        Args:
            filename (PathLike): Name of the file to be saved.
            write_metadata (bool): If true, write the json file with metadata
                (not applicable for dicom files). Currently not used.
            file_format (str | None): Format of the rt dose saved. If None,
                infer it from filename.
            reference_image_path (PathLike | None): Path of the reference dicom image.
                Ignored when saving in formats other than dicom.
        """
        warnings.warn(
            "'Dose.write_image' is deprecated and will be removed in a future release. "
            "Use 'Dose.write' instead."
        )
        return self.write(filename=filename, write_metadata=write_metadata, file_format=file_format)

    def resample(
        self,
        new_spacing: Iterable,
        interpolator: int = sitk.sitkLinear,
        default_pixel_value: float = 0,
    ) -> Dose:
        """Resample the dose with a new voxel spacing (in mm).

        Args:
            new_spacing (Iterable): New voxel spacing of the resampled dose (x, y, z) in mm.
            interpolator (int): Interpolation method used for image resampling.
            default_pixel_value (float): Default value for pixel intensity.

        Returns:
            Dose: Resampled image.
        """
        return Dose(
            super().resample(
                new_spacing=new_spacing,
                interpolator=interpolator,
                default_pixel_value=default_pixel_value,
            )
        )

    def reorient(
        self,
        new_direction: np.ndarray | tuple[float, ...] | None = None,
        *,
        interpolator: int = sitk.sitkBSpline,
        default_pixel_value: float = 0,
        allow_reflection: bool = False,
        flip_axis: int = 0,
    ) -> Image:
        """Rotate the dose to match a target direction cosine matrix.

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
            default_pixel_value (float): Fill value for voxels outside the
                original image extent, forwarded to ``rotate``.
            allow_reflection (bool): Allow reflections for rotations with
                negative determinant. If set to false, improper rotations
                raise an exception.
            flip_axis (int): Internal reflection axis for improper rotations.

        Returns:
            Dose: Reoriented dose matching ``new_direction``.
        """
        return Dose(
            super().reorient(
                new_direction=new_direction,
                interpolator=interpolator,
                default_pixel_value=default_pixel_value,
                allow_reflection=allow_reflection,
                flip_axis=flip_axis,
            )
        )

    def _flip(self, axis: int) -> Dose:
        """Reverse the voxel order along one image axis, mirroring the dose.

        The voxels are reversed along the given axis while the direction and
        origin are updated so the image occupies the same physical extent as
        before; only its handedness changes (the determinant of the direction
        cosine matrix flips sign). The operation is lossless and is its own
        inverse: flipping twice along the same axis restores the original image.

        Args:
            axis (int): Image axis to reverse, in (x, y, z) order
                (``x=0``, ``y=1``, ``z=2``).

        Returns:
            Dose: Mirrored dose with the same spacing, size and metadata, and
                a direction cosine matrix of opposite determinant.
        """
        return Dose(super()._flip(axis=axis))

    def rotate(
        self,
        angle_x: float = 0,
        angle_y: float = 0,
        angle_z: float = 0,
        *,
        interpolator: int = sitk.sitkBSpline,
        default_pixel_value: float = 0,
    ) -> Dose:
        """Apply a 3D rotation to the dose and return the transformed copy.

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
            Dose: Rotated dose with updated direction and origin, and the
                same size, spacing, and metadata as the original.
        """
        return Dose(
            super().rotate(
                angle_x=angle_x,
                angle_y=angle_y,
                angle_z=angle_z,
                interpolator=interpolator,
                default_pixel_value=default_pixel_value,
            )
        )

    def pad(self, reference_image: Image, **kwargs) -> Dose:
        """Pad the dose on top of another image.

        Uses the same notation as ``numpy.pad``.
        The dose is shifted aligning its top-left voxel with the reference image.
        The two images must have the same voxel spacing.
        The shifted dose is cropped if it extends out of the reference image.

        Args:
            reference_image (Image): Image used as reference for padding.
            **kwargs: same arguments used in ``np.pad``.

        Returns:
            Dose: New dose with same shape and spacing of the reference.
        """
        return Dose(super().pad(reference_image=reference_image, **kwargs))

    def coregister(
        self,
        *args,
        **kwargs,
    ) -> Dose:
        """Coregiser the dose on top of another (reference) image.

        Raises:
            NotImplementedError: Coregistration of doses is not supported.
        """
        raise NotImplementedError

    def __add__(self, value: int | float) -> Dose:
        """Add constant value to dose pixel data.

        Args:
            value (int | float): Value to be added to pixel data.

        Returns:
            Dose: Dose with constant value added to pixel data.
        """
        return Dose(super().__add__(value))

    def __sub__(self, value: int | float) -> Dose:
        """Subtract constant value to dose pixel data.

        Args:
            value (int | float): Value to be subtracted to pixel data.

        Returns:
            Dose: Dose with constant value subtracted to pixel data.
        """
        return Dose(super().__sub__(value))

    def __mul__(self, value: int | float) -> Dose:
        """Multiply constant value to dose pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            Dose: Dose with constant value multiplied to pixel data.
        """
        return Dose(super().__mul__(value))

    def __truediv__(self, value: int | float) -> Dose:
        """Divide constant value to dose pixel data.

        Args:
            value (int | float): Value to be multiplied to pixel data.

        Returns:
            Dose: Dose with constant value multiplied to pixel data.
        """
        return Dose(super().__truediv__(value))
