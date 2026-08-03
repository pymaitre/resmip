"""Test module for attributes.py."""

import pytest
from pydicom import Dataset, Sequence

from resmip.dicom_utils.attributes import _copy_type_1, _copy_type_2, _copy_type_3

ZERO_LENGTH_KEYWORDS = ["AccessionNumber", "FrameOfReferenceUID", "SliceThickness"]
"""Keywords covering SH, UI and DS, whose zero-length forms are not represented alike."""


def reference_with_sequence() -> Dataset:
    """Build a reference dataset holding a single-item sequence."""
    item = Dataset()
    item.SeriesInstanceUID = "1.2.3.4"
    reference_dataset = Dataset()
    reference_dataset.ReferencedSeriesSequence = Sequence([item])
    return reference_dataset


def test_copy_type_3_sets_value_when_present_in_reference():
    """A Type 3 attribute present in the reference is copied."""
    reference_dataset = Dataset()
    reference_dataset.AccessionNumber = "ACC123"
    dataset = Dataset()

    _copy_type_3(dataset, reference_dataset, "AccessionNumber")

    assert dataset.AccessionNumber == "ACC123"


def test_copy_type_3_omits_attribute_when_absent_from_reference():
    """A Type 3 attribute absent from the reference is not written."""
    reference_dataset = Dataset()
    dataset = Dataset()

    _copy_type_3(dataset, reference_dataset, "AccessionNumber")

    assert "AccessionNumber" not in dataset


def test_copy_type_2_sets_value_when_present_in_reference():
    """A Type 2 attribute present in the reference is copied."""
    reference_dataset = Dataset()
    reference_dataset.AccessionNumber = "ACC123"
    dataset = Dataset()

    _copy_type_2(dataset, reference_dataset, "AccessionNumber")

    assert dataset.AccessionNumber == "ACC123"


@pytest.mark.parametrize("keyword", ZERO_LENGTH_KEYWORDS)
def test_copy_type_2_writes_zero_length_when_absent_from_reference(keyword: str):
    """A Type 2 attribute absent from the reference becomes zero-length, never omitted."""
    reference_dataset = Dataset()
    dataset = Dataset()

    _copy_type_2(dataset, reference_dataset, keyword)

    assert dataset[keyword].is_empty


def test_copy_type_2_writes_zero_length_when_reference_value_is_empty():
    """A Type 2 attribute that is zero-length in the reference stays zero-length."""
    reference_dataset = Dataset()
    reference_dataset.AccessionNumber = None
    dataset = Dataset()

    _copy_type_2(dataset, reference_dataset, "AccessionNumber")

    assert dataset["AccessionNumber"].is_empty


def test_copy_type_1_sets_value_when_present_in_reference():
    """A Type 1 attribute present in the reference is copied."""
    reference_dataset = Dataset()
    reference_dataset.FrameOfReferenceUID = "1.2.3.4"
    dataset = Dataset()

    _copy_type_1(dataset, reference_dataset, "FrameOfReferenceUID")

    assert dataset.FrameOfReferenceUID == "1.2.3.4"


def test_copy_type_1_raises_when_absent_from_reference():
    """A Type 1 attribute absent from the reference raises instead of being omitted."""
    reference_dataset = Dataset()
    dataset = Dataset()

    with pytest.raises(KeyError):
        _copy_type_1(dataset, reference_dataset, "FrameOfReferenceUID")


def test_copy_type_1_raises_when_reference_value_is_empty():
    """A Type 1 attribute that is zero-length in the reference raises."""
    reference_dataset = Dataset()
    reference_dataset.FrameOfReferenceUID = None
    dataset = Dataset()

    with pytest.raises(KeyError):
        _copy_type_1(dataset, reference_dataset, "FrameOfReferenceUID")


def test_copy_type_1_does_not_share_sequence_with_reference():
    """A copied Type 1 sequence is independent of the reference."""
    reference_dataset = reference_with_sequence()
    dataset = Dataset()
    _copy_type_1(dataset, reference_dataset, "ReferencedSeriesSequence")

    dataset.ReferencedSeriesSequence[0].SeriesInstanceUID = "9.9.9.9"

    assert reference_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID == "1.2.3.4"


def test_copy_type_2_does_not_share_sequence_with_reference():
    """A copied Type 2 sequence is independent of the reference."""
    reference_dataset = reference_with_sequence()
    dataset = Dataset()
    _copy_type_2(dataset, reference_dataset, "ReferencedSeriesSequence")

    dataset.ReferencedSeriesSequence[0].SeriesInstanceUID = "9.9.9.9"

    assert reference_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID == "1.2.3.4"


def test_copy_type_3_does_not_share_sequence_with_reference():
    """A copied Type 3 sequence is independent of the reference."""
    reference_dataset = reference_with_sequence()
    dataset = Dataset()
    _copy_type_3(dataset, reference_dataset, "ReferencedSeriesSequence")

    dataset.ReferencedSeriesSequence[0].SeriesInstanceUID = "9.9.9.9"

    assert reference_dataset.ReferencedSeriesSequence[0].SeriesInstanceUID == "1.2.3.4"
