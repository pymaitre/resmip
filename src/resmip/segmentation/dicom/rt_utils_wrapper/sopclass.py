"""Service-Object Pair (SOP) class functions."""

from pydicom.uid import PYDICOM_IMPLEMENTATION_UID


class SOPClassUID:
    """Identified for the SOP class.

    For more information see here:
    https://dicom.innolitics.com/ciods/rt-dose/sop-common/00080016
    """

    RTSTRUCT_IMPLEMENTATION_CLASS = PYDICOM_IMPLEMENTATION_UID  # TODO find out if this is ok
    DETACHED_STUDY_MANAGEMENT = "1.2.840.10008.3.1.2.3.1"
