"""Utilities for image coregistration."""

from enum import Enum, auto


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
