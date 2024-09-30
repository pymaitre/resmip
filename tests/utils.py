"""Functions used by more than one test module."""

from pathlib import Path


def dicom_ct_path() -> Path:
    """Path of the test Dicom CT."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "CT_00000"


def dicom_rtst_path() -> Path:
    """Path of the test Dicom RTst."""
    return Path(__file__).parent / "Dicom" / "IBSI1_CT_phantom" / "RTst_00000" / "DCM_RS_00060.dcm"


def ibsi_rtst_path() -> Path:
    """Path of the IBSI-compliant RT Structure mask."""
    return Path(__file__).parent / "Nifti" / "IBSI2_CT_phantom" / "mask" / "GTV-1.nii"
