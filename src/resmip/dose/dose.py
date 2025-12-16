"""RT Dose."""

from __future__ import annotations

import logging
from pathlib import Path

import pydicom
import pydicom.errors

from resmip.image import Image
from resmip.utils import PathLike

logger = logging.getLogger(__name__)


class Dose(Image):
    """RT Dose (wrapper of resmip.Image)."""

    @classmethod
    def read_image(
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

        :param filename: Name of the file. If filename ends with ".dcm",
            the reader assumes to read a Dicom rtdose. Otherwise, it assumes a metatadata
            file with the following format exists: f".{filename.stem}.json".
        :type filename: PathLike
        :param read_metadata: If true, read the json file with metadata
            (not applicable for dicom files). Currently not used.
        :type read_metadata: bool
        :param reference_image: 3D image used as reference for dicom Doses, in case
            origin and/or spacing differ. When set to None, a warning is raised and no
            shift / resampling is applied.
        :type reference_image: Image | None
        :return: RT Dose.
        :rtype: Dose
        """
        image = super().read_image(filename=filename, read_metadata=read_metadata)
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

    def write_image(
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

        Currently only non-DICOM file formats are supported

        :param filename: Name of the file to be saved.
        :type filename: PathLike
        :param write_metadata: If true, write the json file with metadata
            (not applicable for dicom files). Currently not used.
        :param file_format: Format of the rt dose saved. If None,
            infer it from filename.
        :type file_format: str | None
        :param reference_image_path: Path of the reference dicom image.
            Ignored when saving in formats other than dicom.
        :type reference_image_path: PathLike | None
        """
        filename = Path(filename)
        if file_format is None:
            file_format = filename.suffix
        filename.parent.mkdir(parents=True, exist_ok=True)
        if file_format != ".dcm":
            return self.write_nondicom(filename)
        raise NotImplementedError("Saving to DICOM RT Dose is currently not supported.")

    def __add__(self, value: int | float) -> Dose:
        """Add constant value to dose pixel data.

        :param value: Value to be added to pixel data.
        :type value: int | float
        :return: Dose with constant value added to pixel data.
        :rtype: Dose
        """
        return Dose(super().__add__(value))

    def __sub__(self, value: int | float) -> Dose:
        """Subtract constant value to dose pixel data.

        :param value: Value to be subtracted to pixel data.
        :type value: int | float
        :return: Dose with constant value subtracted to pixel data.
        :rtype: Dose
        """
        return Dose(super().__sub__(value))

    def __mul__(self, value: int | float) -> Dose:
        """Multiply constant value to dose pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: Dose with constant value multiplied to pixel data.
        :rtype: Dose
        """
        return Dose(super().__mul__(value))

    def __truediv__(self, value: int | float) -> Dose:
        """Divide constant value to dose pixel data.

        :param value: Value to be multiplied to pixel data.
        :type value: int | float
        :return: Dose with constant value multiplied to pixel data.
        :rtype: Dose
        """
        return Dose(super().__truediv__(value))
