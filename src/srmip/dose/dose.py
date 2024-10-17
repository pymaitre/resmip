"""RT Dose."""

from __future__ import annotations

import logging
from typing import Optional

from srmip.image import Image
from srmip.utils import PathLike

logger = logging.getLogger(__name__)


class Dose(Image):
    """RT Dose (wrapper of srmip.Image)."""

    @classmethod
    def read_image(
        cls,
        filename: PathLike,
        read_metadata: bool = True,
        reference_image: Optional[Image] = None,
    ) -> Dose:
        """
        Read rt dose from file.

        Doses could have different origin and/or
        spacing compared to the referenced series.
        This function, shifts the dose accordingly.

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
        if reference_image is None:
            logger.warning("No reference image has been provided for the RT Dose.")
            return new_dose
        new_dose = new_dose.resample(new_spacing=reference_image.spacing)
        new_dose = new_dose.pad(reference_image=reference_image)
        return cls(new_dose)
