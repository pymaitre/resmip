"""Segmentation."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

from resmip import Image
from resmip.dicom_utils import string_tag_for_keyword
from resmip.image import DicomModality, ImageDTypeLike
from resmip.rt_structure.utils import get_structure_name_from_filename
from resmip.utils import PathLike

from .dicom.rtstruct import _read as _dicom_rtstruct_read
from .dicom.rtstruct import _write as _dicom_rtstruct_write
from .dicom.seg import _read as _dicom_seg_read
from .dicom.seg import _write as _dicom_seg_write
from .utils import SegmentationType

__all__ = ["RTStructureSet", "Segmentation", "SegmentationCollection"]

logger = logging.getLogger(__name__)


class Segmentation(Image):
    """A segmentation mask associated with an image.

    Segmentations are binary or fractional masks that annotate a region
    of interest on top of an image. They can be read from and written to
    DICOM SEG and DICOM RTSTRUCT files, as well as non-DICOM formats such
    as NIfTI and MetaImage.

    DICOM SEG supports both binary and fractional segmentation types.
    DICOM RTSTRUCT supports binary segmentations only.
    """

    def __init__(
        self,
        *args,
        name: str = "",
        segmentation_type: SegmentationType = SegmentationType.binary,
        **kwargs,
    ):
        """Initialise a Segmentation.

        Args:
            *args: Positional arguments forwarded to ``resmip.Image.__init__``.
            name (str): Name of the segmentation, e.g. the anatomical structure
                label.
            segmentation_type (SegmentationType): Encoding type of the
                segmentation mask. Either ``SegmentationType.binary`` or
                ``SegmentationType.fractional``.
            **kwargs: Keyword arguments forwarded to ``resmip.Image.__init__``.
        """
        if not isinstance(name, str):
            raise ValueError(
                f"type({name}) ({type(name)}) is not a valid type for name. Supported type(s): str."
            )
        super().__init__(*args, modality=DicomModality.seg, **kwargs)
        self._name = name
        self._segmentation_type = segmentation_type

    def __getitem__(self, key) -> Segmentation:
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
        return Segmentation(
            super().__getitem__(key), name=self.name, segmentation_type=self.segmentation_type
        )

    @property
    def name(self) -> str:
        """Name of the Segmentation."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def modality(self) -> str:
        """DICOM modality of the Segmentation.

        Only "SEG" is supported.
        """
        image_modality = self._get_modality()
        if image_modality != DicomModality.seg.value:
            raise ValueError(f"{image_modality} is not a valid modality value for Segmentations.")
        return image_modality

    def astype(self, dtype: ImageDTypeLike) -> Segmentation:
        """Return a copy of the segmentation with the pixel array cast to a new dtype.

        Args:
            dtype (ImageDTypeLike): Target dtype for the pixel array. If ``None``,
                the default dtype defined in ``FORMAT_TO_TYPESTR`` is used.

        Returns:
            Segmentation: New segmentation with the same name and segmentation
                type, with pixel data cast to ``dtype``.
        """
        return Segmentation(
            super().astype(dtype=dtype), name=self.name, segmentation_type=self.segmentation_type
        )

    @property
    def segmentation_type(self) -> SegmentationType:
        """Encoding type of the segmentation mask.

        Either ``SegmentationType.binary`` or ``SegmentationType.fractional``.
        """
        return self._segmentation_type

    @segmentation_type.setter
    def segmentation_type(self, value):
        self._segmentation_type = value

    def _infer_segmentation_type(self):
        """Infer and update the segmentation type from the pixel array.

        If more than two unique voxel values are found, the segmentation
        type is changed to ``SegmentationType.fractional``. This method
        mutates the segmentation in place.
        """
        if len(np.unique(self.numpy(copy=False))) > 2:
            logger.info("Changing segmentation type of %s to fractional", self)
            self.segmentation_type = SegmentationType.fractional

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
        segmentation_type: SegmentationType = SegmentationType.binary,
        **kwargs,
    ) -> Segmentation:
        """Create a segmentation from a numpy array and spatial metadata.

        Args:
            array (np.ndarray): 3D voxel array with axis order (z, y, x).
            spacing (tuple[float, float, float]): Voxel spacing in mm (x, y, z).
            origin (tuple[float, float, float]): World coordinates of the
                first voxel in mm (x, y, z).
            direction (tuple[float]): Flattened row-major direction cosine
                matrix (9 elements).
            metadata (dict[str, str] | None): DICOM or sidecar metadata
                key-value pairs.
            name (str): Name of the segmentation.
            segmentation_type (SegmentationType): Encoding type of the mask.
            **kwargs: Additional keyword arguments forwarded to
                ``resmip.Image.__init__``.

        Returns:
            Segmentation: New segmentation with the specified spatial metadata.
        """
        return Segmentation(
            super().from_array(
                array=array,
                spacing=spacing,
                origin=origin,
                direction=direction,
                metadata=metadata,
                **kwargs,
            ),
            name=name,
            segmentation_type=segmentation_type,
        )

    @classmethod
    def read(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        structure_name: str | None = None,
        reference_image: Image | None = None,
    ) -> Segmentation:
        """Read a single segmentation from a file.

        The file format is inferred from the path. DICOM files (``.dcm``)
        can contain either a SEG or RTSTRUCT dataset. Non-DICOM files
        (e.g. ``.nii.gz``, ``.mha``) are read directly.

        Args:
            filename (PathLike): Path to the segmentation file. Files with
                a ``.dcm`` suffix are read as DICOM SEG or RTSTRUCT.
                Other suffixes are read as non-DICOM image files, with
                an optional sidecar JSON metadata file named
                ``.{stem}.json``.
            read_metadata (bool): If ``True``, read the sidecar JSON metadata
                file for non-DICOM formats. Has no effect for DICOM files.
            structure_name (str | None): Name of the segmentation to extract.
                Required for DICOM files. For non-DICOM files,
                if ``None`` the name is inferred from the filename.
            reference_image (Image | None): Reference image used to define
                the spatial grid when reading DICOM RTSTRUCT files.
                Required for RTSTRUCT; ignored for all other formats.

        Raises:
            ValueError: If reading an RTSTRUCT and ``structure_name`` or
                ``reference_image`` is not provided. If reading a SEG and
                ``structure_name`` is not provided.
            KeyError: If no matching segment is found in the DICOM file.
            NotImplementedError: If the DICOM modality is not SEG or RTSTRUCT.

        Returns:
            Segmentation: The segmentation read from the file.
        """
        filename = Path(filename)
        if filename.suffix == ".dcm":
            return Segmentation._read_dicom(
                filename=filename, structure_name=structure_name, reference_image=reference_image
            )
        return Segmentation._read_nondicom(
            filename=filename, read_metadata=read_metadata, structure_name=structure_name
        )

    @classmethod
    def _read_dicom(
        cls, filename: Path, structure_name: str | None = None, reference_image: Image | None = None
    ):
        """Read a single segmentation from a DICOM SEG or RTSTRUCT file.

        Determines the modality from the DICOM header and delegates to
        the appropriate reader.

        Args:
            filename (Path): Path to the DICOM file.
            structure_name (str | None): Name of the segment or ROI to
                extract.
            reference_image (Image | None): Reference image defining the
                spatial grid. Required for RTSTRUCT.

        Raises:
            ValueError: If reading an RTSTRUCT and ``structure_name`` or
                ``reference_image`` is not provided. If reading a SEG and
                ``structure_name`` is not provided.
            KeyError: If no matching segment or ROI is found.
            NotImplementedError: If the DICOM modality is not SEG or RTSTRUCT.

        Returns:
            Segmentation: The matched segmentation.
        """
        if structure_name is None:
            raise ValueError("Must specify a structure name for dicom segmentations.")
        segmentation_dataset = pydicom.dcmread(filename, force=True)
        dataset_modality = segmentation_dataset.Modality
        if dataset_modality == "SEG":
            seg_coll = SegmentationCollection.read(
                filename=filename, structure_names=[structure_name], regex=False
            )
            try:
                return next(iter(seg_coll.values()))
            except StopIteration:
                raise KeyError("No matching structure found.")  # pylint: disable=raise-missing-from
        if dataset_modality == "RTSTRUCT":
            if reference_image is None:
                raise ValueError("Must specify a reference image for dicom RT Structures.")
            rtst = RTStructureSet.read(
                filename=filename,
                structure_names=[structure_name],
                reference_image=reference_image,
                regex=False,
            )
            try:
                return next(iter(rtst.values()))
            except StopIteration:
                raise KeyError("No matching structure found.")  # pylint: disable=raise-missing-from
        raise NotImplementedError(
            f"{dataset_modality} is an unsupported modality for segmentations."
        )

    @classmethod
    def _read_nondicom(cls, filename, read_metadata=True, structure_name=None):
        """Read a segmentation from a non-DICOM image file.

        The segmentation type is inferred from the pixel array: if more
        than two unique voxel values are found, the type is set to
        ``SegmentationType.fractional``.

        Args:
            filename (PathLike): Path to the image file.
            read_metadata (bool): If ``True``, read sidecar JSON metadata
                from ``.{stem}.json`` if present.
            structure_name (str | None): Name to assign to the segmentation.
                If ``None``, the name is inferred from the filename.

        Returns:
            Segmentation: Segmentation with name and type inferred from
                the file.
        """
        if structure_name is None:
            structure_name = get_structure_name_from_filename(filename)
        new_segmentation = Segmentation(
            super()._read_nondicom(filename, read_metadata), name=structure_name
        )
        if len(np.unique(new_segmentation)) > 2:
            new_segmentation.segmentation_type = SegmentationType.fractional
        return new_segmentation

    def write(
        self,
        filename: PathLike,
        *,
        write_metadata: bool = True,
        use_existing_ids: bool = False,
        modality: str | DicomModality = DicomModality.seg,
        reference_image_path: PathLike | None = None,
        series_description: str | None = None,
    ):
        """Write the segmentation to a file.

        The output format is inferred from the filename suffix. For DICOM
        output (``.dcm``), the target modality determines whether a SEG or
        RTSTRUCT file is produced. For non-DICOM formats, the image is
        written directly and metadata is optionally saved to a sidecar JSON.

        Args:
            filename (PathLike): Path to the output file. A ``.dcm`` suffix
                produces a DICOM file; other suffixes produce non-DICOM files.
            write_metadata (bool): If ``True``, write a sidecar JSON metadata
                file alongside non-DICOM outputs. Has no effect for DICOM.
            use_existing_ids (bool): Reserved for future use.
            modality (str | DicomModality): Target DICOM modality. Either
                ``DicomModality.seg`` (default) or ``DicomModality.rtstruct``.
                Only used when writing to a ``.dcm`` file.
            reference_image_path (PathLike | None): Path to the reference
                DICOM series directory used to populate patient and study
                metadata. Required for DICOM output.
            series_description (str | None): Value for the ``SeriesDescription``
                tag in the output DICOM file.

        Raises:
            ValueError: If ``modality`` is not a supported DICOM modality
                for segmentations.
        """
        filename = Path(filename)
        if filename.suffix == ".dcm":
            if isinstance(modality, DicomModality):
                modality = modality.value
            if modality == DicomModality.rtstruct.value:
                return RTStructureSet([self]).write(
                    filename=filename,
                    reference_image_path=reference_image_path,
                    series_description=series_description,
                )
            if modality == DicomModality.seg.value:
                return SegmentationCollection([self]).write(
                    filename=filename,
                    reference_image_path=reference_image_path,
                    series_description=series_description,
                )
            raise ValueError(
                f"Unsupported modality for segmentations: {modality}. Supported "
                f"values are: {DicomModality.seg.value}, {DicomModality.rtstruct.value}"
            )
        return self._write_nondicom(filename=filename, write_metadata=write_metadata)

    def resample(
        self,
        new_spacing: Iterable,
        interpolator: int = sitk.sitkNearestNeighbor,
        default_pixel_value: float = 0,
    ) -> Segmentation:
        """Resample the segmentation to a new voxel spacing.

        Nearest-neighbour interpolation is used by default to preserve
        binary mask values. Using any other interpolator will produce a
        warning, as it may introduce non-binary voxel values.

        Args:
            new_spacing (Iterable): Target voxel spacing in mm (x, y, z).
            interpolator (int): SimpleITK interpolator constant.
            default_pixel_value (float): Value used for voxels outside the
                original image extent after resampling.

        Returns:
            Segmentation: Resampled segmentation with the same name and
                segmentation type.
        """
        if interpolator != sitk.sitkNearestNeighbor:
            logger.warning(
                "Only sitk.sitkNearestNeighbor should be used when resampling binary segmentations."
            )
        resampled_image = super().resample(
            new_spacing=new_spacing,
            interpolator=interpolator,
            default_pixel_value=default_pixel_value,
        )
        resampled_structure = Segmentation(
            resampled_image, name=self.name, segmentation_type=self.segmentation_type
        )
        return resampled_structure

    def pad(self, reference_image: Image, **kwargs) -> Segmentation:
        """Pad the segmentation to match the spatial extent of a reference image.

        The segmentation is shifted so that its first voxel aligns with the
        first voxel of the reference image. Both images must have the same
        voxel spacing. Regions of the segmentation that extend beyond the
        reference image extent are cropped.

        Args:
            reference_image (Image): Image defining the target spatial extent.
            **kwargs: Additional keyword arguments forwarded to ``numpy.pad``.

        Returns:
            Segmentation: New segmentation with the same size, spacing, and
                origin as ``reference_image``.
        """
        return Segmentation(
            super().pad(reference_image=reference_image, **kwargs),
            name=self.name,
            segmentation_type=self.segmentation_type,
        )

    def coregister(self, *args, **kwargs) -> Segmentation:
        """Coregiser the segmentation on top of another (reference) image.

        Raises:
            NotImplementedError: Coregistration of segmentations is not supported.
        """
        raise NotImplementedError

    def to_binary(self, threshold: int | float = 1) -> Segmentation:
        """Convert the segmentation to a binary mask.

        Voxels with values greater than or equal to ``threshold`` are set
        to 1; all others are set to 0. The output pixel array is cast to
        ``uint8``.

        Args:
            threshold (int | float): Binarisation threshold. Must be
                positive. Defaults to ``1``.

        Raises:
            ValueError: If ``threshold`` is less than or equal to zero.

        Returns:
            Segmentation: New binary segmentation with the same spatial
                metadata and name as the original.
        """
        if threshold <= 0:
            raise ValueError(f"The provided threshold ({threshold}) is <= 0.")
        array = self.numpy()
        array = (array >= threshold).astype(np.uint8)
        return Segmentation.from_array(
            array=array,
            spacing=self.spacing,
            origin=self.origin,
            direction=self.direction,
            metadata=self.metadata,
            name=self.name,
            segmentation_type=SegmentationType.binary,
        )

    def is_compatible(self, other: Segmentation) -> bool:
        """Check whether this segmentation is spatially compatible with another.

        Two segmentations are compatible if they share the same size,
        spacing, direction, and origin. Incompatible attributes are logged
        as warnings.

        Args:
            other (Segmentation): Segmentation to compare against.

        Returns:
            bool: ``True`` if both segmentations are spatially compatible.
        """
        checks = ["size", "spacing", "direction", "origin"]
        for attribute in checks:
            if getattr(self, attribute) != getattr(other, attribute):
                logger.warning(
                    "The two segmentations have different %s: %s, %s.",
                    attribute,
                    getattr(self, attribute),
                    getattr(other, attribute),
                )
                return False
        return True

    def __add__(self, value: int | float) -> Segmentation:
        """Add a constant to the segmentation pixel data.

        Args:
            value (int | float): Constant to add to every voxel.

        Returns:
            Segmentation: New segmentation with ``value`` added to pixel data.
        """
        return Segmentation(
            super().__add__(value), name=self.name, segmentation_type=self.segmentation_type
        )

    def __sub__(self, value: int | float) -> Segmentation:
        """Subtract a constant from the segmentation pixel data.

        Args:
            value (int | float): Constant to subtract from every voxel.

        Returns:
            Segmentation: New segmentation with ``value`` subtracted from pixel data.
        """
        return Segmentation(
            super().__sub__(value), name=self.name, segmentation_type=self.segmentation_type
        )

    def __mul__(self, value: int | float) -> Segmentation:
        """Multiply the segmentation pixel data by a constant.

        Args:
            value (int | float): Constant to multiply every voxel by.

        Returns:
            Segmentation: New segmentation with pixel data multiplied by ``value``.
        """
        return Segmentation(
            super().__mul__(value), name=self.name, segmentation_type=self.segmentation_type
        )

    def __truediv__(self, value: int | float) -> Segmentation:
        """Divide the segmentation pixel data by a constant.

        Args:
            value (int | float): Constant to divide every voxel by.

        Returns:
            Segmentation: New segmentation with pixel data divided by ``value``.
        """
        return Segmentation(
            super().__truediv__(value), name=self.name, segmentation_type=self.segmentation_type
        )


class SegmentationCollection(dict[str, Segmentation]):
    """An ordered collection of named segmentations.

    Behaves as a dictionary keyed by segmentation name. All segmentations
    in a collection must have unique names. Methods that operate on spatial
    properties (``size``, ``spacing``, ``direction``) require all
    segmentations to be spatially compatible.
    """

    def __init__(self, segmentations: Iterable[Segmentation] | None = None):
        """Initialise the collection from an iterable of segmentations.

        Args:
            segmentations (Iterable[Segmentation] | None): Segmentations to
                add. If ``None``, an empty collection is created.

        Raises:
            KeyError: If any two segmentations share the same name.
        """
        if segmentations is None:
            segmentations = []
        segmentation_names = [seg.name for seg in segmentations]
        if len(set(segmentation_names)) != len(segmentation_names):
            raise KeyError(f"Duplicate names were found in segmentations: {segmentation_names}.")
        segmentation_map = {seg.name: seg for seg in segmentations}
        self.update(segmentation_map)

    @classmethod
    def from_rt_structure_set(cls, rtst: RTStructureSet) -> SegmentationCollection:
        """Create a collection from an RT Structure Set.

        Args:
            rtst (RTStructureSet): Source RT structure set. All segmentations
                are included without conversion.

        Returns:
            SegmentationCollection: New collection containing the same
                segmentations as ``rtst``.
        """
        return cls(rtst.values())

    def append(self, segmentation: Segmentation, ignore_errors: bool = False):
        """Add a segmentation to the collection.

        Args:
            segmentation (Segmentation): Segmentation to add.
            ignore_errors (bool): If ``True``, silently skip adding the
                segmentation when its name already exists. If ``False``,
                raise a ``KeyError`` instead.

        Raises:
            KeyError: If a segmentation with the same name already exists
                and ``ignore_errors`` is ``False``.
        """
        if segmentation.name in self:
            if not ignore_errors:
                raise KeyError(
                    f"Segmentation with name {segmentation.name} already "
                    "present in the collection."
                )
            logger.warning(
                "Segmentation with name %s already "
                "present in the collection. Not adding the segmentation.",
                segmentation.name,
            )
            return
        self.update({segmentation.name: segmentation})

    def extend(self, segmentations: Iterable[Segmentation], ignore_errors: bool = False):
        """Add multiple segmentations to the collection.

        Args:
            segmentations (Iterable[Segmentation]): Segmentations to add.
            ignore_errors (bool): Forwarded to ``append``. If ``True``,
                duplicate names are silently skipped.

        Raises:
            KeyError: If any segmentation name already exists in the collection
                and ``ignore_errors`` is ``False``.
        """
        for seg in segmentations:
            self.append(seg, ignore_errors=ignore_errors)

    def rename(self, current_name: str, new_name: str):
        """Rename a segmentation in the collection.

        Args:
            current_name (str): Existing name of the segmentation to rename.
            new_name (str): New name to assign.

        Raises:
            KeyError: If ``current_name`` is not present in the collection,
                or if ``new_name`` already exists (and differs from
                ``current_name``).
        """
        if current_name not in self:
            raise KeyError(
                f"Segmentation name '{current_name}' not present "
                f"in the collection. Current names are {self.keys()}."
            )
        if new_name in self:
            if current_name == new_name:
                logger.warning("Renaming the structure %s with the same name.", current_name)
                return
            raise KeyError(
                f"New segmentation name '{new_name}' already exists "
                f"in the collection. Current names are {self.keys()}."
            )
        renamed_segmentation = self.pop(current_name)
        renamed_segmentation.name = new_name
        self[new_name] = renamed_segmentation

    @property
    def modality(self) -> str:
        """DICOM modality of the SegmentationCollection.

        Only "SEG" is supported.
        """
        return DicomModality.seg.value

    @classmethod
    def read(
        cls,
        filename: PathLike | list[PathLike],
        *,
        structure_names: list[str] | None = None,
        read_metadata: bool = True,
        regex: bool = False,
        parallel: bool = True,
    ) -> SegmentationCollection:
        """Read a segmentation collection from a DICOM SEG file or a list of non-DICOM files.

        Args:
            filename (PathLike | list[PathLike]): Path to a DICOM SEG file,
                or a list of paths to non-DICOM image files (e.g. NIfTI).
            structure_names (list[str] | None): Names of the segments to
                read. If ``None``, all segments are returned. Only used
                for DICOM SEG input.
            read_metadata (bool): If ``True``, read the sidecar JSON metadata
                file for non-DICOM formats. Has no effect for DICOM files.
            regex (bool): If ``True``, ``structure_names`` are treated as
                regular expression patterns.
            parallel (bool): If ``True``, convert segments to images in
                parallel using a thread pool.

        Returns:
            SegmentationCollection: Collection of segmentations read from
                the file or files.
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            segmentations, metadata = _dicom_seg_read(
                filename,
                structure_names=structure_names,
                regex=regex,
                parallel=parallel,
            )
            segmentation_type = SegmentationType(
                metadata[string_tag_for_keyword("SegmentationType")]
            )
            return cls(
                [
                    Segmentation(
                        seg.image,
                        name=seg.name,
                        metadata=metadata,
                        segmentation_type=segmentation_type,
                    )
                    for seg in segmentations
                    if seg.name is not None
                ]
            )
        return cls._read_nondicom(filename=filename, read_metadata=read_metadata)

    @classmethod
    def _read_nondicom(cls, filename: list[PathLike], read_metadata: bool = True):
        """Read segmentations from a list of non-DICOM files.

        Args:
            filename (list[PathLike]): List of paths to non-DICOM image files.
            read_metadata (bool): If ``True``, read the sidecar JSON metadata
                file for non-DICOM formats.

        Returns:
            SegmentationCollection: Collection of segmentations read from
                the provided files.
        """
        segmentations = []
        for f in filename:
            segmentations.append(Segmentation.read(f, read_metadata=read_metadata))
        return cls(segmentations)

    def write(
        self,
        filename: PathLike | list[PathLike],
        *,
        file_format: str | None = None,
        write_metadata: bool = True,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:
        """Write the segmentation collection to a file or directory.

        For DICOM output, all segmentations are written to a single
        ``.dcm`` file. For non-DICOM output, each segmentation is written
        to a separate file, either to the paths provided in ``filename``
        or auto-named within a directory using ``file_format`` as the suffix.

        Args:
            filename (PathLike | list[PathLike]): Output path. A single
                path with a ``.dcm`` suffix produces a DICOM SEG file.
                A directory path requires ``file_format`` to be set.
                A list of paths writes one non-DICOM file per segmentation.
            file_format (str | None): File suffix for non-DICOM output when
                ``filename`` is a directory (e.g. ``".nii.gz"``). Must be
                provided when writing to a directory.
            write_metadata (bool): If ``True``, write a sidecar JSON metadata
                file alongside non-DICOM outputs. Has no effect for DICOM.
            reference_image_path (PathLike | None): Path to the reference
                DICOM series directory. Required for DICOM SEG output.
                Ignored for non-DICOM formats.
            series_description (str): Value for the ``SeriesDescription``
                tag in DICOM SEG output. Ignored for non-DICOM formats.
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            if not filename.is_dir():
                return _dicom_seg_write(
                    self, filename, reference_image_path, series_description=series_description
                )
            if file_format is None:
                raise ValueError(
                    "File format must be specified when "
                    "saving a non-DICOM segmentation collection to a directory."
                )
            filename = [filename / f"{struct_name}{file_format}" for struct_name in self]
        return self._write_nondicom(filename=filename, write_metadata=write_metadata)

    def _write_nondicom(self, filename: list[PathLike], write_metadata: bool = True):
        """Write each segmentation in the collection to a separate non-DICOM file.

        Args:
            filename (list[PathLike]): List of output file paths, one per
                segmentation. Must have the same length as the collection.
            write_metadata (bool): If ``True``, write a sidecar JSON metadata
                file alongside non-DICOM outputs.

        Raises:
            ValueError: If the number of filenames does not match the number
                of segmentations.
        """
        filename = [Path(f) for f in filename]
        if len(filename) != len(self):
            raise ValueError(
                "The number of filenames provided is different than the number of structures."
            )
        for structure_filename, structure in zip(filename, self.values()):
            structure.write(structure_filename, write_metadata=write_metadata)

    def validate(self):
        """Check that all segmentations in the collection are spatially compatible.

        Raises:
            ValueError: If any two segmentations differ in size, spacing,
                direction, or origin.
        """
        previous = None
        for seg in self.values():
            if previous is not None:
                if not previous.is_compatible(seg):
                    raise ValueError("Not all segmentations are compatible.")
            previous = seg

    @property
    def _first(self) -> Segmentation | None:
        """Return first element of the collection to retrieve properties."""
        self.validate()
        return next(iter(self.values()), None)

    @property
    def spacing(self) -> tuple[float, float, float] | None:
        """Voxel spacing shared by all segmentations in mm (x, y, z).

        Raises:
            ValueError: If the segmentations are not spatially compatible.

        Returns:
            tuple[float, float, float] | None: Common voxel spacing, or
                ``None`` if the collection is empty.
        """
        return self._first.spacing if self._first else None

    @property
    def direction(self) -> tuple[float, ...] | None:
        """Direction cosine matrix shared by all segmentations.

        Raises:
            ValueError: If the segmentations are not spatially compatible.

        Returns:
            tuple[float, ...] | None: Flattened 9-element direction cosine
                matrix, or ``None`` if the collection is empty.
        """
        return self._first.direction if self._first else None

    @property
    def size(self) -> tuple[int, int, int] | None:
        """Image size shared by all segmentations (x, y, z).

        Raises:
            ValueError: If the segmentations are not spatially compatible.

        Returns:
            tuple[int, int, int] | None: Common image dimensions, or
                ``None`` if the collection is empty.
        """
        return self._first.size if self._first else None

    @property
    def segmentation_type(self) -> SegmentationType:
        """Segmentation type of the collection.

        Infers the type of each segmentation from its pixel data before
        checking. Returns ``SegmentationType.fractional`` if any segmentation
        is fractional; otherwise returns ``SegmentationType.binary``.
        Returns ``SegmentationType.binary`` for empty collections.
        """
        for seg in self.values():
            seg._infer_segmentation_type()  # pylint: disable=protected-access
            if seg.segmentation_type == SegmentationType.fractional:
                return SegmentationType.fractional
        return SegmentationType.binary


class RTStructureSet(SegmentationCollection):
    """A collection of binary segmentations stored as a DICOM RT Structure Set.

    Extends ``SegmentationCollection`` with RTSTRUCT-specific read/write
    behaviour. All segmentations must be binary; fractional masks are not
    supported by the RTSTRUCT format.
    """

    def __init__(self, segmentations: Iterable[Segmentation] | None = None):
        """Initialise the RT Structure Set.

        Args:
            segmentations (Iterable[Segmentation] | None): Segmentations to
                include. If ``None``, an empty structure set is created.
        """
        super().__init__(segmentations=segmentations)

    @classmethod
    def from_segmentation_collection(cls, collection: SegmentationCollection) -> RTStructureSet:
        """Create an RT Structure Set from a segmentation collection.

        Each segmentation is converted to binary before being added.

        Args:
            collection (SegmentationCollection): Source collection of
                segmentations.

        Returns:
            RTStructureSet: New structure set containing binary versions
                of all segmentations in ``collection``.
        """
        rtst = cls()
        for seg in collection.values():
            rtst.append(segmentation=seg.to_binary(), ignore_errors=False)
        return rtst

    @property
    def modality(self) -> str:
        """DICOM modality of the RT Structure Set.

        Only "RTSTRUCT" is supported.
        """
        return DicomModality.rtstruct.value

    @classmethod
    def read(
        cls,
        filename: PathLike | list[PathLike],
        *,
        structure_names: list[str] | None = None,
        read_metadata: bool = True,
        regex: bool = False,
        parallel: bool = True,
        reference_image: Image | None = None,
    ) -> RTStructureSet:
        """Read an RT Structure Set from a DICOM RTSTRUCT file or non-DICOM files.

        Args:
            filename (PathLike | list[PathLike]): Path to a DICOM RTSTRUCT
                file, or a list of paths to non-DICOM image files.
            structure_names (list[str] | None): Names of the ROIs to read.
                If ``None``, all ROIs are returned. Only used for DICOM
                RTSTRUCT input.
            read_metadata (bool): If ``True``, read the sidecar JSON metadata
                file for non-DICOM formats. Has no effect for DICOM files.
            regex (bool): If ``True``, ``structure_names`` are treated as
                regular expression patterns. Defaults to ``False``.
            parallel (bool): If ``True``, convert ROIs to images in parallel
                using a thread pool.
            reference_image (Image | None): Reference image defining the
                spatial grid onto which contours are rasterised. Required
                for DICOM RTSTRUCT input.

        Returns:
            RTStructureSet: Structure set containing the matched ROIs.
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            segmentations = _dicom_rtstruct_read(
                filename,
                reference_image=reference_image,
                structure_names=structure_names,
                regex=regex,
                parallel=parallel,
            )
            return cls(
                [Segmentation(x.image, name=x.name) for x in segmentations if x.name is not None]
            )
        return cls._read_nondicom(filename=filename, read_metadata=read_metadata)

    def write(
        self,
        filename: PathLike | list[PathLike],
        *,
        file_format: str | None = None,
        write_metadata: bool = True,
        reference_image_path: PathLike | None = None,
        series_description: str = "",
    ) -> None:
        """Write the RT Structure Set to a DICOM RTSTRUCT file or non-DICOM files.

        For DICOM output, all ROIs are written to a single ``.dcm`` file.
        For non-DICOM output, each segmentation is written to a separate
        file.

        Args:
            filename (PathLike | list[PathLike]): Output path. A single path
                with a ``.dcm`` suffix produces a DICOM RTSTRUCT file.
                A directory path requires ``file_format`` to be set.
                A list of paths writes one non-DICOM file per segmentation.
            file_format (str | None): File suffix for non-DICOM output when
                ``filename`` is a directory (e.g. ``".nii.gz"``). Must be
                provided when writing to a directory.
            write_metadata (bool): If ``True``, write a sidecar JSON metadata
                file alongside non-DICOM outputs.
            reference_image_path (PathLike | None): Path to the reference
                DICOM series directory. Required for DICOM RTSTRUCT output.
                Ignored for non-DICOM formats.
            series_description (str): Value for the ``SeriesDescription``
                tag in DICOM RTSTRUCT output. Ignored for non-DICOM formats.

        Raises:
            ValueError: If ``filename`` is a directory and ``file_format``
                is not specified.
        """
        if isinstance(filename, PathLike.__args__):
            filename = Path(filename)
            if not filename.is_dir():
                return _dicom_rtstruct_write(
                    self, filename, reference_image_path, series_description=series_description
                )
            if file_format is None:
                raise ValueError(
                    "File format must be specified when "
                    "saving a non-DICOM RT structure set to a directory."
                )
            filename = [filename / f"{struct_name}{file_format}" for struct_name in self]
        return self._write_nondicom(filename=filename, write_metadata=write_metadata)
