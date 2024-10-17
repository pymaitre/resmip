"""Image object containing pixel array and metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

import srmip.dicom_utils.series as dicom_series
from srmip import DICOM_FIELDS
from srmip.image._data_types import ImageDTypeLike, _sitk_image_dtype
from srmip.utils import PathLike, format_digit_string


class Image(sitk.Image):
    """Wrapper class of SimpleITK.Image with support to headers."""

    _metadata: Dict[str, str]
    """Dictionary containing metadata."""

    def __init__(self, *args):
        """Call sitk.Image constructor and create an empty dictionary for the header."""
        super().__init__(*args)
        self._metadata = {}
        # Copy metadata when creating an image from an existing one
        if len(args) > 0:
            if isinstance(args[0], Image):
                self._metadata = args[0].metadata

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

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        spacing: Tuple[float],
        origin: Tuple[float],
        direction: Tuple[float],
        metadata: Optional[Dict[str, str]] = None,
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
        new_image = cls(sitk.GetImageFromArray(array))
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

    def __array__(self, dtype: Optional[Union[str, npt.DTypeLike]] = None) -> np.ndarray:
        """
        Convert an image to a numpy array.

        Wrapper of sitk.GetArrayFromImage().

        :param dtype: The dtype to use for the numpy array.
            If None, the default dtype of the image is used
            as defined the global `FORMAT_TO_TYPESTR` dictionary.
        :type dtype: str | npt.DTypeLike | None
        :return: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        :rtype: np.ndarray
        """
        image_array = sitk.GetArrayFromImage(self)
        if dtype is not None:
            image_array = image_array.astype(dtype)
        return image_array

    def numpy(self, dtype: Optional[Union[str, npt.DTypeLike]] = None) -> np.ndarray:
        """
        Generate a numpy array of pixels from the image.

        Wrapper of sitk.GetArrayFromImage().

        :param dtype: The dtype to use for the numpy array.
            If None, the default dtype of the image is used
            as defined the global `FORMAT_TO_TYPESTR` dictionary.
        :type dtype: str | npt.DTypeLike | None
        :return: Image array as numpy array of shape (z_dim, y_dim, x_dim).
        :rtype: np.ndarray
        """
        return self.__array__(dtype=dtype)

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

    def write_image(self, filename: PathLike, write_metadata: bool = True) -> None:
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
