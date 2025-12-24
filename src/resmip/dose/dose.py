"""RT Dose."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable

import pydicom
import pydicom.errors
import SimpleITK as sitk

from resmip.image import Image, ImageDTypeLike
from resmip.image.metadata import DicomModality
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
        write_metadata: bool = False,
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
            return self._write_nondicom(filename)
        raise NotImplementedError("Saving to DICOM RT Dose is currently not supported.")

    def write_image(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = False,
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
