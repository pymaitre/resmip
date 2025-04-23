"""Image object containing pixel array and metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

import srmip.dicom_utils.series as dicom_series
from srmip import DICOM_FIELDS
from srmip.image._data_types import ImageDTypeLike, _is_unsigned, _sitk_image_dtype
from srmip.utils import PathLike, format_digit_string

logger = logging.getLogger(__name__)


class Image(sitk.Image):
    """Wrapper class of SimpleITK.Image with support to headers."""

    def __init__(self, *args):
        """Call sitk.Image constructor and create an empty dictionary for the header."""
        super().__init__(*args)
        self._metadata = {}
        """Dictionary containing metadata."""
        # Copy metadata when creating an image from an existing one
        if len(args) > 0:
            if isinstance(args[0], Image):
                self._metadata = args[0].metadata

    def __getitem__(self, key) -> Image:
        """
        Get a pixel value, a sliced image, or a metadata item.

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
        return Image(super().__getitem__(key))

    @property
    def metadata(self) -> Dict[str, str]:
        """Dicom header combined with other metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @property
    def spacing(self) -> Tuple[float]:
        """Voxel spacing in mm (x, y, z)."""
        return self.GetSpacing()

    @spacing.setter
    def spacing(self, value: Tuple[float]):
        self.SetSpacing(value)
        # add the spacing to metadata too
        self.metadata[DICOM_FIELDS["PixelSpacing"]] = "\\".join([str(x) for x in value[:2]])
        self.metadata[DICOM_FIELDS["SliceThickness"]] = str(value[2])

    @property
    def origin(self) -> Tuple[float]:
        """Coordinates of the top left voxel in mm (x, y, z)."""
        return self.GetOrigin()

    @origin.setter
    def origin(self, value: Tuple[float]):
        """The original DICOM header key is not updated."""
        self.SetOrigin(value)

    @property
    def direction(self) -> Tuple[float]:
        """
        Direction cosine matrix.

        For more information, see here:
        https://dicom.innolitics.com/ciods/rt-dose/image-plane/00200037
        """
        return self.GetDirection()

    @direction.setter
    def direction(self, value: Tuple[float]):
        self.SetDirection(value)
        # add the spacing to metadata too (only xy direction)
        self.metadata[DICOM_FIELDS["ImageOrientationPatient"]] = "\\".join(
            [str(x) for x in value[:-3]]
        )

    @property
    def size(self) -> Tuple[float]:
        """Image size in pixels."""
        return self.GetSize()

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        *,
        spacing: Tuple[float],
        origin: Tuple[float],
        direction: Tuple[float],
        metadata: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Image:
        """
        Create a new image from a numpy array.

        :param array: 3D array containing voxel values for the image (z, y, x).
        :type array: np.ndarray
        :param spacing: Voxel spacing for the image in mm (x, y, z).
        :type spacing: tuple[float]
        :param origin: Coordinates of the top left voxel in mm (x, y, z).
        :type origin: tuple[float]
        :param direction: Direction cosine matrix.
        :type direction: tuple[float]
        :param metadata: Metadata containing information from the DICOM header.
        :type metadata: Optional[Dict[str, str]]
        :return: New image
        :rtype: Image
        """
        new_image = cls(sitk.GetImageFromArray(array), **kwargs)
        if metadata is not None:
            new_image.metadata = metadata
        new_image.spacing = spacing
        new_image.origin = origin
        new_image.direction = direction
        return new_image

    @staticmethod
    def metadata_file_name(filename: PathLike) -> Path:
        """
        Generate the filename for the metadata.

        Defaults a json file with same name of the output image file (filename).
        The json filename is prepended with a "." to make it hidden.

        :param filename: name of the output image file name.
        :type filename: PathLike
        :return: Path of the json metadata file.
        :rtype: Path
        """
        filename = Path(filename)
        return filename.parent / f".{filename.stem}.json"

    def __array__(
        self, dtype: Optional[Union[str, npt.DTypeLike]] = None, view: bool = False
    ) -> np.ndarray:
        """
        Convert an image to a numpy array.

        Wrapper of sitk.GetArrayFromImage().

        :param dtype: The dtype to use for the numpy array.
            If None, the default dtype of the image is used
            as defined the global `FORMAT_TO_TYPESTR` dictionary.
        :type dtype: str | npt.DTypeLike | None
        :param view: If set to true, return a view of the underlying data,
            without copying them. If a dtype is specified, a copy is returned anyway.
        :type view: bool
        :return: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        :rtype: np.ndarray
        """
        if view:
            image_array = sitk.GetArrayViewFromImage(self)
        else:
            image_array = sitk.GetArrayFromImage(self)
        if dtype is not None:
            image_array = image_array.astype(dtype)
        return image_array

    def numpy(
        self, dtype: Optional[Union[str, npt.DTypeLike]] = None, view: bool = False
    ) -> np.ndarray:
        """
        Generate a numpy array of pixels from the image.

        Wrapper of sitk.GetArrayFromImage().

        :param dtype: The dtype to use for the numpy array.
            If None, the default dtype of the image is used
            as defined the global `FORMAT_TO_TYPESTR` dictionary.
        :type dtype: str | npt.DTypeLike | None
        :param view: If set to true, return a view of the underlying data,
            without copying them. If a dtype is specified, a copy is returned anyway.
        :type view: bool
        :return: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        :rtype: np.ndarray
        """
        return self.__array__(dtype=dtype, view=view)

    def astype(self, dtype: ImageDTypeLike) -> Image:
        """
        Convert pixel array type to the specified value, by casting a new image.

        :param dtype: The dtype to use for the numpy array.
            If None, the default dtype of the image is used
            as defined the global `FORMAT_TO_TYPESTR` dictionary.
        :type dtype: ImageDTypeLike
        :return: new image with specified data type.
        :rtype: Image
        """
        new_image = Image(sitk.Cast(self, _sitk_image_dtype(dtype)))
        new_image.metadata = self.metadata
        return new_image

    @classmethod
    def read_image(cls, filename: PathLike, read_metadata: bool = True) -> Image:
        """
        Load image file (and metadata).

        The image format is automatically determined from filename's suffix.

        :param filename: Name of the file. If filename is a directory,
            the reader assumes to read a Dicom series. Otherwise, it assumes a metatadata
            file with the following format exists: f".{filename.stem}.json".
        :type filename: PathLike
        :param read_metadata: If true, read the json file with metadata
            (not applicable for dicom files).
        :type read_metadata: bool
        :return: Image and metadata.
        :rtype: Image
        """
        filename = Path(filename)
        if filename.is_dir():
            sitk_image, series_metadata = dicom_series.read(filename)
        else:
            sitk_image = sitk.ReadImage(filename)
            if read_metadata and Image().metadata_file_name(filename).exists():
                serialized_metadata = Image().metadata_file_name(filename).read_text()
                series_metadata = json.loads(serialized_metadata)
            else:
                series_metadata = {}
            for key in sitk_image.GetMetaDataKeys():
                value = sitk_image.GetMetaData(key)
                value = format_digit_string(value)
                series_metadata[key] = value
        new_image = cls(sitk_image)
        new_image.metadata = series_metadata
        return new_image

    def write_image(self, filename: PathLike, *, write_metadata: bool = True) -> None:
        """
        Save image file (and metadata).

        The image format is automatically determined from filename's suffix.
        If parent directories of filename do not exist, they are created.

        :param filename: Name of the file. If filename is a directory,
            the writer assumes to write a Dicom series.
        :type filename: PathLike
        :param write_metadata: If true, save the json file with metadata
            (not applicable for dicom files).
        :type write_metadata: bool
        """
        filename = Path(filename)
        if not filename.exists() and filename.suffix == "":
            filename.mkdir(parents=True, exist_ok=True)
        if filename.is_dir():
            dicom_series.write(self, self.metadata, filename)
            return
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.write_nondicom(filename=filename, write_metadata=write_metadata)

    def write_nondicom(self, filename: PathLike, write_metadata: bool = True) -> None:
        """
        Save image file (and metadata) to non-DICOM formats using ITK.

        :param filename: Name of the file. If filename is a directory,
            the writer assumes to write a Dicom series.
        :type filename: PathLike
        :param write_metadata: If true, save the json file with metadata
            (not applicable for dicom files).
        :type write_metadata: bool
        """
        sitk.WriteImage(self, filename)
        if write_metadata:
            serialized_metadata = json.dumps(self.metadata)
            self.metadata_file_name(filename).write_text(serialized_metadata)

    def resample(
        self,
        new_spacing: Iterable,
        interpolator: int = sitk.sitkLinear,
        default_pixel_value: float = 0,
    ) -> Image:
        """
        Resample the image with a new voxel spacing (in mm).

        :param new_spacing: New voxel spacing of the resampled image (x, y, z) in mm.
        :type new_spacing: Iterable
        :param interpolator: Interpolation method used for image resampling.
        :type interpolator: int
        :param default_pixel_value: Default value for pixel intensity.
        :type default_pixel_value: float
        :return: Resampled image.
        :rtype: Image
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
        new_img.metadata[DICOM_FIELDS["PixelSpacing"]] = "\\".join(
            [str(x) for x in new_img.spacing[:2]]
        )
        new_img.metadata[DICOM_FIELDS["SliceThickness"]] = str(new_img.spacing[2])
        return new_img

    def pad(self, reference_image: Image, **kwargs) -> Image:
        """
        Pad the image on top of another image.

        Uses the same notation as `numpy.pad`.
        The image is shifted aligning its top-left voxel with the reference image.
        The two images must have the same voxel spacing.
        The shifted image is cropped if it extends out of the reference image.

        :param reference_image: Image used as reference for padding.
        :type reference_image: Image
        :return: New image with same shape and spacing of the reference.
        :rtype: Image
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
        )

    @staticmethod
    def _get_coregistration_method(
        seed: int = 0, num_threads: Optional[int] = None
    ) -> sitk.ImageRegistrationMethod:
        """
        Generate method used for coregistration.

        :param seed: Random seed for the registration method. When set to 0,
            uses system walltime. Use different values for deterministic behaviour.
        :type seed: int
        :param num_threads: Number of threads used for coregistration. By default, it is
            set to the maximum number of available threads.
            Set it to 1 for deterministic behaviour.
        :type num_threads: Optional[int]
        :return: Registration method used for coregistration.
        :rtype: sitk.ImageRegistrationMethod
        """
        registration_method = sitk.ImageRegistrationMethod()
        if num_threads:
            registration_method.SetGlobalDefaultNumberOfThreads(num_threads)
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01, seed=seed)
        registration_method.SetInterpolator(sitk.sitkLinear)
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=200,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        return registration_method

    def coregister(
        self,
        reference_image: Image,
        fill_value: float = 0.0,
        seed: int = 0,
        num_threads: Optional[int] = None,
    ) -> Image:
        """
        Coregiser the image on top of another (reference) image.

        :param reference_image: Image used as reference for coregistration.
        :type reference_image: Image
        :param fill_value: Value used to fill voxels during resampling (defaults to 0).
        :type fill_value: float
        :param seed: Random seed for the registration method. When set to 0,
            uses system walltime. Use different values for deterministic behaviour.
        :type seed: int
        :param num_threads: Number of threads used for coregistration. By default, it is
            set to the maximum number of available threads.
            Set it to 1 for deterministic behaviour.
        :type num_threads: Optional[int]
        :return: New image coregistered with reference_image.
        :rtype: Image
        """
        registration_method = self._get_coregistration_method(seed=seed, num_threads=num_threads)
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
            )
        )
        moving_image = moving_image.astype(current_type)
        moving_image.metadata = self.metadata
        return moving_image

    def __add__(self, value: Union[int, float]) -> Image:
        """
        Add constant value to pixel data.

        :param value: Value to be added to pixel data.
        :type value: int | float
        :return: Image with constant value added to pixel data.
        :rtype: Image
        """
        current_image = self
        if isinstance(value, float):
            logger.debug("Casting image type to float.")
            current_image = self.astype(float)
        if value < 0:
            if _is_unsigned(current_image.GetPixelID()):
                logger.warning(
                    "Adding a negative value when image "
                    "type is unsigned, make sure to cast it to a signed type "
                    "if pixel values become negative."
                )
        transformed_image = Image(super(Image, current_image).__add__(value))
        transformed_image.metadata = self.metadata
        return transformed_image

    def __sub__(self, value: Union[int, float]) -> Image:
        """
        Subtract constant value to pixel data.

        :param value: Value to be subtracted to pixel data.
        :type value: int | float
        :return: Image with constant value subtracted to pixel data.
        :rtype: Image
        """
        current_image = self
        if isinstance(value, float):
            logger.debug("Casting image type to float")
            current_image = self.astype(float)
        if _is_unsigned(current_image.GetPixelID()):
            logger.warning(
                "Subtracting value when image "
                "type is unsigned, make sure to cast it to a signed type "
                "if pixel values become negative."
            )
        transformed_image = Image(super(Image, current_image).__sub__(value))
        transformed_image.metadata = self.metadata
        return transformed_image

    def __mul__(self, value: Union[int, float]) -> Image:
        """
        Multiply constant value to pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: Image with constant value multiplied to pixel data.
        :rtype: Image
        """
        current_image = self
        if isinstance(value, float):
            logger.debug("Casting image type to float")
            current_image = self.astype(float)
        if value < 0:
            if _is_unsigned(current_image.GetPixelID()):
                logger.warning(
                    "Multipying a negative value when image "
                    "type is unsigned, make sure to cast it to a signed type "
                    "if pixel values become negative."
                )
        transformed_image = Image(super(Image, current_image).__mul__(value))
        transformed_image.metadata = self.metadata
        return transformed_image

    def __truediv__(self, value: Union[int, float]) -> Image:
        """
        Multiply constant value to pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: Image with constant value multiplied to pixel data.
        :rtype: Image
        """
        logger.debug("Casting image type to float")
        current_image = self.astype(float)
        transformed_image = Image(super(Image, current_image).__truediv__(value))
        transformed_image.metadata = self.metadata
        return transformed_image
