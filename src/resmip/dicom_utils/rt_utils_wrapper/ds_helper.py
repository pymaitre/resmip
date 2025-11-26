"""Wrapper module from rt-utils."""

import datetime

from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from resmip.dicom_utils.rt_utils_wrapper.sopclass import SOPClassUID


def generate_base_dataset() -> FileDataset:
    """Generate the FileDataset used for the DICOM RTst."""
    file_name = "rt-utils-struct"
    file_meta = get_file_meta()
    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)
    add_required_elements_to_ds(ds)
    add_sequence_lists_to_ds(ds)
    return ds


def get_file_meta() -> FileMetaDataset:
    """Generate file meta information."""
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SOPClassUID.RTSTRUCT
    file_meta.MediaStorageSOPInstanceUID = generate_uid()  # TODO find out random generation is fine
    file_meta.ImplementationClassUID = SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS
    return file_meta


def add_required_elements_to_ds(ds: FileDataset):
    """Add basic information to the DICOM header."""
    dt = datetime.datetime.now()
    # Append data elements required by the DICOM standarad
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    ds.StructureSetLabel = "RTstruct"
    ds.StructureSetDate = dt.strftime("%Y%m%d")
    ds.StructureSetTime = dt.strftime("%H%M%S.%f")
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = "Qurit"
    ds.ManufacturerModelName = "rt-utils"
    ds.InstitutionName = "Qurit"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.ApprovalStatus = "UNAPPROVED"


def add_sequence_lists_to_ds(ds: FileDataset):
    """Generate sequences for contours."""
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()


def add_patient_information(ds: FileDataset, series_data: list[Dataset]):
    """Add patient information read from the reference image to the header."""
    reference_ds = series_data[0]
    ds.PatientName = getattr(reference_ds, "PatientName", "")
    ds.PatientID = getattr(reference_ds, "PatientID", "")
    ds.PatientBirthDate = getattr(reference_ds, "PatientBirthDate", "")
    ds.PatientSex = getattr(reference_ds, "PatientSex", "")
    ds.PatientAge = getattr(reference_ds, "PatientAge", "")
    ds.PatientSize = getattr(reference_ds, "PatientSize", "")
    ds.PatientWeight = getattr(reference_ds, "PatientWeight", "")


def add_refd_frame_of_ref_sequence(ds: FileDataset, series_data: list[Dataset]):
    """Set frame of reference for the rtst equal to the one from the reference image."""
    refd_frame_of_ref = Dataset()
    refd_frame_of_ref.FrameOfReferenceUID = getattr(
        series_data[0], "FrameOfReferenceUID", generate_uid()
    )
    refd_frame_of_ref.RTReferencedStudySequence = create_frame_of_ref_study_sequence(series_data)

    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence.append(refd_frame_of_ref)


def create_frame_of_ref_study_sequence(series_data: list[Dataset]) -> Sequence:
    """Set frame of reference from the referenced study."""
    reference_ds = series_data[0]
    rt_refd_series = Dataset()
    rt_refd_series.SeriesInstanceUID = reference_ds.SeriesInstanceUID
    rt_refd_series.ContourImageSequence = create_contour_image_sequence(series_data)

    rt_refd_series_sequence = Sequence()
    rt_refd_series_sequence.append(rt_refd_series)

    rt_refd_study = Dataset()
    rt_refd_study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
    rt_refd_study.ReferencedSOPInstanceUID = reference_ds.StudyInstanceUID
    rt_refd_study.RTReferencedSeriesSequence = rt_refd_series_sequence

    rt_refd_study_sequence = Sequence()
    rt_refd_study_sequence.append(rt_refd_study)
    return rt_refd_study_sequence


def create_contour_image_sequence(series_data: list[Dataset]) -> Sequence:
    """Set frame of reference from the referenced series."""
    contour_image_sequence = Sequence()

    for series in series_data:
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = series.SOPClassUID
        contour_image.ReferencedSOPInstanceUID = series.SOPInstanceUID
        contour_image_sequence.append(contour_image)

    return contour_image_sequence
