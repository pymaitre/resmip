"""Image object containing pixel array and metadata."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from srmip.dicom_nifti_conversion.series import read_dicom_series, write_dicom_series
from srmip.utils import PathLike, format_digit_string


class Image(sitk.Image):
    """Wrapper class of SimpleITK.Image with support to headers."""

    _metadata: Dict[str, str]
    """Dictionary containing metadata."""

    def __init__(self, *args):
        """Call sitk.Image constructor and create an empty dictionary for the header."""
        super().__init__(*args)
        self._metadata = {}

    @property
    def metadata(self) -> Dict[str, str]:
        """Dicom header combined with other metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

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
        :rtype: np.array
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
        :rtype: np.array
        """
        return self.__array__(dtype=dtype)

    @staticmethod
    def read_image(filename: PathLike, read_metadata: bool = True) -> Image:
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
            sitk_image, series_metadata = read_dicom_series(filename)
        else:
            # new_image = Image(sitk.ReadImage(filename))
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
        new_image = Image(sitk_image)
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
            write_dicom_series(self, self.metadata, filename)
            return
        filename.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self, filename)
        if write_metadata:
            serialized_metadata = json.dumps(self.metadata)
            self.metadata_file_name(filename).write_text(serialized_metadata)
