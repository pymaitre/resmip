"""Test module for segmentation collection objects."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from pydicom import Dataset, dcmread

from resmip.image import DicomModality
from resmip.segmentation import (
    RTStructureSet,
    Segmentation,
    SegmentationCollection,
    SegmentationType,
)

from .utils import (
    fractional_highdicom_dicom_seg,
    fractional_liver_dicom_seg,
    liver_dicom_seg,
    overlap_highdicom_dicom_seg,
    singleframe_highdicom_dicom_seg,
)


def assert_required_tags_in_dicom_seg(dataset: Dataset):
    """Check if all required fields for DICOM SEG are present."""
    assert dataset.file_meta.MediaStorageSOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert dataset.file_meta.TransferSyntaxUID is not None

    assert dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert dataset.SOPInstanceUID is not None
    assert dataset.Modality == "SEG"
    assert dataset.Rows > 0
    assert dataset.Columns > 0
    assert int(dataset.NumberOfFrames) > 0
    assert dataset.BitsAllocated in (1, 8)
    assert dataset.BitsStored in (1, 8)
    assert dataset.PixelRepresentation == 0
    assert dataset.SegmentationType in ("BINARY", "FRACTIONAL")
    assert dataset.ContentLabel is not None


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_init_empty_collection(collection_type):
    """Generate empty ``SegmentationCollection`` or ``RTStructureSet``."""
    collection = collection_type()
    assert type(collection) == collection_type
    assert len(collection) == 0


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_init_two_segmentations(collection_type):
    """Generate collection with two different segmentations."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type([segm_a, segm_b])
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["a", "b"]
    assert collection["a"] == segm_a
    assert collection["b"] == segm_b


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_init_two_segmentations_same_name(collection_type):
    """Generate collection with two different segmentations with the same name.

    It should return a ``KeyError``.
    """
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "a"
    with pytest.raises(KeyError):
        _ = collection_type([segm_a, segm_b])


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_append_new_segmentation(collection_type):
    """Append a segmentation to the collection."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type([segm_a])
    assert type(collection) == collection_type
    assert len(collection) == 1
    collection.append(segm_b)
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["a", "b"]
    assert collection["a"] == segm_a
    assert collection["b"] == segm_b


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
@pytest.mark.parametrize("ignore_errors", [True, False])
def test_append_new_segmentation_same_name(collection_type, ignore_errors):
    """Append a segmentation to a collection already containing the segmentation."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "a"
    collection = collection_type([segm_a])
    assert type(collection) == collection_type
    assert len(collection) == 1
    if ignore_errors is False:
        with pytest.raises(KeyError):
            collection.append(segm_b, ignore_errors=ignore_errors)
        return
    collection.append(segm_b, ignore_errors=ignore_errors)
    assert type(collection) == collection_type
    assert len(collection) == 1
    for key in collection:
        assert key in ["a"]
    assert collection["a"] == segm_a


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_append_new_segmentation(collection_type):
    """Extend two segmentations to the collection."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type()
    assert type(collection) == collection_type
    collection.extend([segm_a, segm_b])
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["a", "b"]
    assert collection["a"] == segm_a
    assert collection["b"] == segm_b


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
@pytest.mark.parametrize("ignore_errors", [True, False])
def test_extend_new_segmentation_same_name(collection_type, ignore_errors):
    """Append two segmentations to a collection already containing one segmentation."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type([segm_a])
    assert type(collection) == collection_type
    assert len(collection) == 1
    if ignore_errors is False:
        with pytest.raises(KeyError):
            collection.extend([segm_a, segm_b], ignore_errors=ignore_errors)
        return
    collection.extend([segm_a, segm_b], ignore_errors=ignore_errors)
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["a", "b"]
    assert collection["a"] == segm_a
    assert collection["b"] == segm_b


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_rename_collection(collection_type):
    """Rename one of the segmentations in the collection."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type([segm_a, segm_b])
    assert type(collection) == collection_type
    collection.rename("a", "c")
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["c", "b"]
    assert collection["c"] == segm_a
    assert collection["c"].name == "c"
    assert collection["b"] == segm_b


def test_rename_collection_nonexisting_key():
    """Rename a nonexisting segmentation in the collection."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = SegmentationCollection([segm_a, segm_b])
    with pytest.raises(KeyError):
        collection.rename("c", "d")


def test_rename_collection_already_existing_key():
    """Rename one of the segmentations in the collection with an already existing name."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = SegmentationCollection([segm_a, segm_b])
    with pytest.raises(KeyError):
        collection.rename("a", "b")


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_rename_collection_same_key(collection_type):
    """Rename one of the segmentations in the collection with the same name."""
    segm_a = MagicMock(name="segm_a")
    segm_a.name = "a"
    segm_b = MagicMock(name="segm_b")
    segm_b.name = "b"
    collection = collection_type([segm_a, segm_b])
    assert type(collection) == collection_type
    collection.rename("a", "a")
    assert type(collection) == collection_type
    assert len(collection) == 2
    for key in collection:
        assert key in ["a", "b"]
    assert collection["a"] == segm_a
    assert collection["b"] == segm_b


def test_create_rtst_from_segmentation_collection():
    segm_a = Segmentation(name="a", segmentation_type=SegmentationType.binary)
    segm_b = Segmentation(name="b", segmentation_type=SegmentationType.fractional)
    collection = SegmentationCollection([segm_a, segm_b])
    segm_types = [seg.segmentation_type for seg in collection.values()]
    assert SegmentationType.fractional in segm_types
    rtst = RTStructureSet.from_segmentation_collection(collection)
    segm_types = [seg.segmentation_type for seg in rtst.values()]
    assert SegmentationType.fractional not in segm_types
    assert rtst.modality == DicomModality.rtstruct.value


def test_create_segmentation_collection_rtst_from():
    segm_a = Segmentation(name="a", segmentation_type=SegmentationType.binary)
    segm_b = Segmentation(name="b", segmentation_type=SegmentationType.fractional)
    collection = SegmentationCollection([segm_a, segm_b])
    segm_types = [seg.segmentation_type for seg in collection.values()]
    assert SegmentationType.fractional in segm_types
    rtst = RTStructureSet.from_segmentation_collection(collection)
    segm_types = [seg.segmentation_type for seg in rtst.values()]
    assert SegmentationType.fractional not in segm_types
    coll = SegmentationCollection.from_rt_structure_set(rtst)
    segm_types = [seg.segmentation_type for seg in coll.values()]
    assert SegmentationType.fractional not in segm_types
    assert coll.modality == DicomModality.seg.value


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
@pytest.mark.parametrize("segmentations_number", [0, 1, 2])
def test_validate_valid_segmentation_collection(collection_type, segmentations_number):
    mask = np.linspace(0, 10, 300).reshape((3, 10, 10))
    dummy_segmentation = Segmentation.from_array(
        mask,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        name="dummy",
        segmentation_type=SegmentationType.fractional,
    )
    other_segmentation = Segmentation.from_array(
        mask * 2,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        name="other_dummy",
        segmentation_type=SegmentationType.fractional,
    )
    segs = [dummy_segmentation, other_segmentation]
    coll = collection_type(segs[:segmentations_number])
    assert len(coll) == segmentations_number
    coll.validate()
    assert type(coll) == collection_type


@pytest.mark.parametrize("collection_type", [SegmentationCollection, RTStructureSet])
def test_validate_invalid_segmentation_collection(collection_type):
    mask = np.linspace(0, 10, 300).reshape((3, 10, 10))
    dummy_segmentation = Segmentation.from_array(
        mask,
        spacing=(1, 1, 1),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        name="dummy",
        segmentation_type=SegmentationType.fractional,
    )
    other_segmentation = Segmentation.from_array(
        mask,
        spacing=(1, 2, 1),
        origin=(0, 0, 0),
        direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
        name="other_dummy",
        segmentation_type=SegmentationType.fractional,
    )
    coll = collection_type([dummy_segmentation, other_segmentation])
    assert len(coll) == 2
    with pytest.raises(ValueError):
        coll.validate()


def test_read_liver_dicom_segmentation():
    """Read a binary DICOM SEG."""
    seg = Segmentation.read(filename=liver_dicom_seg(), structure_name="Liver")
    assert seg.patient_id == "99000"
    assert seg.modality == DicomModality.seg.value
    assert seg.size == (512, 512, 3)
    assert seg.name == "Liver"
    assert seg.segmentation_type == SegmentationType.binary
    assert seg.numpy().sum() == 107098
    reference = Segmentation.read(liver_dicom_seg().parent / "Liver.nii.gz")
    assert np.all(seg.numpy(copy=False) == reference.numpy(copy=False))


def test_read_liver_dicom_segmentation_collection():
    """Read a binary DICOM SEG."""
    seg = SegmentationCollection.read(filename=liver_dicom_seg())
    assert len(seg) == 1
    assert "Liver" in seg

    assert seg["Liver"].patient_id == "99000"
    assert seg["Liver"].origin == (-235.2, -226.8, -128.69)
    assert seg["Liver"].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert seg["Liver"].spacing == (0.810547, 0.810547, 1.0)
    assert seg["Liver"].segmentation_type == SegmentationType.binary


def test_read_fractional_liver_dicom_segmentation_collection():
    """Read a fractional DICOM SEG."""
    seg = SegmentationCollection.read(filename=fractional_liver_dicom_seg())
    assert len(seg) == 1
    assert "Liver" in seg

    assert seg["Liver"].patient_id == "99000"
    assert seg["Liver"].origin == (-235.2, -226.8, -128.69)
    assert seg["Liver"].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert seg["Liver"].spacing == (0.810547, 0.810547, 1.0)
    assert seg["Liver"].segmentation_type == SegmentationType.fractional


@pytest.mark.parametrize("parallel", [True, False])
def test_read_highdicom_overlap_dicom_segmentation_collection(parallel):
    """Read a fractional DICOM SEG from ``highdicom``."""
    seg = SegmentationCollection.read(filename=overlap_highdicom_dicom_seg(), parallel=parallel)
    assert len(seg) == 2
    for key in ["first segment", "second segment"]:
        assert key in seg
        assert seg[key].patient_id == "77654033"
        assert np.allclose(seg[key].origin, (-125.0, -128.1, 101.77))
        assert seg[key].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
        assert seg[key].spacing == (0.488281, 0.488281, 1.25)
        assert seg[key].segmentation_type != SegmentationType.fractional


def test_read_highdicom_fractional_dicom_segmentation_collection():
    """Read a fractional DICOM SEG from ``highdicom``."""
    seg = SegmentationCollection.read(filename=fractional_highdicom_dicom_seg())
    assert len(seg) == 1
    assert "first segment" in seg

    assert seg["first segment"].patient_id == "77654033"
    assert np.allclose(seg["first segment"].origin, (-125.0, -128.1, 103.02))
    assert seg["first segment"].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert seg["first segment"].spacing == (0.488281, 0.488281, 1.25)
    assert seg["first segment"].segmentation_type == SegmentationType.fractional


@pytest.mark.xfail(raises=NotImplementedError)
def test_read_highdicom_singleframe_dicom_segmentation_collection():
    """Read a binary single-frame DICOM SEG from ``highdicom``."""
    seg = SegmentationCollection.read(filename=singleframe_highdicom_dicom_seg())
    assert len(seg) == 1
    assert "first segment" in seg

    assert seg["first segment"].patient_id == "77654033"
    assert np.allclose(seg["first segment"].origin, (-125.0, -128.1, 103.02))
    assert seg["first segment"].direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    assert seg["first segment"].spacing == (0.488281, 0.488281, 1.25)
    assert seg["first segment"].segmentation_type == SegmentationType.fractional


@pytest.mark.parametrize(
    "segmentation_type", [SegmentationType.binary, SegmentationType.fractional]
)
@pytest.mark.parametrize("incorrect_segmentation_type", [True, False])
def test_write_liver_dicom_segmentation(
    segmentation_type: SegmentationType, incorrect_segmentation_type: bool, tmp_path
):
    """Write a binary/fractional DICOM SEG."""
    if segmentation_type == SegmentationType.binary:
        dicom_path = liver_dicom_seg()
    elif segmentation_type == SegmentationType.fractional:
        dicom_path = fractional_liver_dicom_seg()
    else:
        raise NotImplementedError
    seg = Segmentation.read(filename=dicom_path, structure_name="Liver")
    assert seg.segmentation_type == segmentation_type
    if incorrect_segmentation_type:
        seg.segmentation_type = SegmentationType.binary
    if segmentation_type == SegmentationType.binary:
        assert set(np.unique(seg.numpy(copy=False))) == set((0, 1))
    else:
        assert set(np.unique(seg.numpy(copy=False))) == set((0, 85, 170, 255))
    output_path = tmp_path / "seg.dcm"
    seg.write(
        output_path, modality=DicomModality.seg, reference_image_path=dicom_path.parent / "liver_ct"
    )

    ds = dcmread(output_path)
    ds.PatientID == seg.patient_id
    assert ds.PatientID == "99000"
    assert ds.Modality == DicomModality.seg.value
    assert seg.size == (512, 512, 3)
    assert seg.name == "Liver"
    assert ds.SegmentationType == segmentation_type.value
    assert np.count_nonzero(seg.numpy()) == 107098
    assert seg.size[2] == ds.NumberOfFrames
    assert seg.size == ds.pixel_array.shape[::-1]
    assert np.all(seg.numpy(copy=False) == ds.pixel_array)
    assert_required_tags_in_dicom_seg(ds)


@pytest.mark.parametrize(
    "segmentation_type", [SegmentationType.binary, SegmentationType.fractional]
)
@pytest.mark.parametrize("incorrect_segmentation_type", [True, False])
def test_write_liver_dicom_segmentation_set(
    segmentation_type: SegmentationType, incorrect_segmentation_type: bool, tmp_path
):
    """Write a binary/fractional DICOM SEG with two identical segments."""
    if segmentation_type == SegmentationType.binary:
        dicom_path = liver_dicom_seg()
    elif segmentation_type == SegmentationType.fractional:
        dicom_path = fractional_liver_dicom_seg()
    else:
        raise NotImplementedError
    seg1 = Segmentation.read(filename=dicom_path, structure_name="Liver")
    seg1.name = "Liver1"
    seg2 = Segmentation.read(filename=dicom_path, structure_name="Liver")
    seg2.name = "Liver2"
    coll = SegmentationCollection([seg1, seg2])
    if incorrect_segmentation_type:
        for key in coll:
            coll[key].segmentation_type = SegmentationType.binary
    coll.validate()
    assert coll.segmentation_type == segmentation_type
    output_path = tmp_path / "seg.dcm"
    coll.write(output_path, reference_image_path=dicom_path.parent / "liver_ct")

    ds = dcmread(output_path)
    ds.PatientID == seg1.patient_id
    assert ds.PatientID == "99000"
    assert ds.Modality == DicomModality.seg.value
    assert seg1.size == (512, 512, 3)
    assert seg1.name == "Liver1"
    assert ds.SegmentationType == segmentation_type.value
    assert np.count_nonzero(seg1.numpy()) == 107098
    assert sum([s.size[2] for s in coll.values()]) == ds.NumberOfFrames
    assert seg1.size[:2] == ds.pixel_array.shape[:0:-1]
    assert seg2.size[:2] == ds.pixel_array.shape[:0:-1]
    assert seg1.size[2] + seg2.size[2] == ds.pixel_array.shape[0]
    assert np.all(seg1.numpy(copy=False) == ds.pixel_array[::2])
    assert np.all(seg2.numpy(copy=False) == ds.pixel_array[1::2])
    assert_required_tags_in_dicom_seg(ds)

    written_seg1 = Segmentation.read(output_path, structure_name="Liver1")
    assert np.all(written_seg1.numpy(copy=False) == seg1.numpy(copy=False))
