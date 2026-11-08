"""Utilities for image coregistration."""

from dataclasses import dataclass
from enum import Enum, auto

__all__ = [
    "AffineRegistration",
    "CoregistrationMetric",
    "RegistrationMethod",
    "RigidRegistration",
]


class CoregistrationMetric(Enum):
    """Metrics used for image coregistration."""

    correlation = auto()
    """Use negative normalized cross correlation image metric.

    It can be useful when the two images have similar pixel
    values (i.e.: same modality).
    More information can be found here:
    https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1CorrelationImageToImageMetricv4.html
    """
    mutual_information = auto()
    """Use the mutual information between two images to be registered.

    Uses the method of Mattes et al.

    It can be useful when the two images with different modalities.
    More information can be found here:
    https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1MattesMutualInformationImageToImageMetricv4.html
    """


class RegistrationMethod:
    """Base type of the registration methods accepted by ``Image.register``."""


@dataclass(frozen=True)
class RigidRegistration(RegistrationMethod):
    """Rigid registration: rotation and translation, six degrees of freedom.

    Suits images of the same patient whose anatomy has not deformed, such as
    an MR acquired on the same day as the planning CT.
    """

    metric: CoregistrationMetric = CoregistrationMetric.mutual_information
    """Similarity metric optimised by the registration."""


@dataclass(frozen=True)
class AffineRegistration(RegistrationMethod):
    """Affine registration: rigid motion plus scaling and shearing, twelve degrees of freedom.

    Suits images whose geometry differs by more than a rigid motion, such as a
    calibration error in voxel size, or a global change in patient size.
    """

    metric: CoregistrationMetric = CoregistrationMetric.mutual_information
    """Similarity metric optimised by the registration."""
