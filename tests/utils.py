"""Functions used by more than one test module."""

from pathlib import Path

import numpy as np

import resmip


def dicom_ct_path() -> Path:
    """Path of the test Dicom CT."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "CT_00000"


def dicom_rtst_path() -> Path:
    """Path of the test Dicom RTst."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "RTst_00000" / "DCM_RS_00060.dcm"


def dicom_rtst_path_with_hole() -> Path:
    """Path of the test Dicom RTst with a hole."""
    return (
        Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "RTst_00001" / "DCM_RS_with_hole.dcm"
    )


def ibsi_rtst_path() -> Path:
    """Path of the IBSI-compliant RT Structure mask."""
    return Path(__file__).parent / "Nifti" / "IBSI2_CT_phantom" / "mask" / "GTV-1.nii"


def coregistered_image_path(coregistration_metric) -> Path:
    """Path of the coregistered CT image used for testing."""
    return Path(__file__).parent / "Nifti" / "coregistered" / f"{coregistration_metric}.nii.gz"


def liver_dicom_seg() -> Path:
    """Path of a binary segmentation saved as DICOM SEG."""
    return Path(__file__).parent / "Dicom" / "liver" / "liver.dcm"


def fractional_liver_dicom_seg() -> Path:
    """Path of a fractional segmentation saved as DICOM SEG."""
    return Path(__file__).parent / "Dicom" / "liver" / "fractional_liver.dcm"


def fractional_highdicom_dicom_seg() -> Path:
    """Path of a fractional segmentation saved as DICOM SEG."""
    return Path(__file__).parent / "Dicom" / "liver" / "seg_image_ct_binary_fractional.dcm"


def overlap_highdicom_dicom_seg() -> Path:
    """Path of a fractional segmentation with two segments saved as DICOM SEG."""
    return Path(__file__).parent / "Dicom" / "liver" / "seg_image_ct_binary_overlap_correct.dcm"


def singleframe_highdicom_dicom_seg() -> Path:
    """Path of a binary single-frame segmentation saved as DICOM SEG."""
    return Path(__file__).parent / "Dicom" / "liver" / "seg_image_ct_binary_single_frame.dcm"


def generate_dummy_segmentation(mask: np.ndarray):
    """Generate dummy segmentation from mask array."""
    return resmip.Segmentation.from_array(
        mask,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        name="dummy",
        segmentation_type=resmip.SegmentationType.fractional,
    )
