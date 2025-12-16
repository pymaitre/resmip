"""Functions used by more than one test module."""

from pathlib import Path


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
