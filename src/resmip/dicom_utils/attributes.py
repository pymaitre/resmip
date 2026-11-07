"""Copy DICOM attributes from a reference dataset according to their requirement type.

Requirement type is a property of the (attribute, module, IOD) triple, never of the
attribute alone. These functions do not know the type of any attribute: the caller
states it by choosing the function, and each applies only the presence logic that
follows from that type.

Values are deep-copied, so that a sequence written into the target dataset does
not share its items with the reference.
"""

from copy import deepcopy

from pydicom import Dataset


def _copy_type_1(dataset: Dataset, reference_dataset: Dataset, keyword: str) -> None:
    """Copy a Type 1 attribute, which must be present with a value.

    Args:
        dataset (Dataset): Target dataset to populate.
        reference_dataset (Dataset): Source dataset to read the attribute from.
        keyword (str): DICOM keyword of the attribute to copy.

    Raises:
        KeyError: If the attribute is absent from the reference dataset, or present
            with a zero-length value. Degrading to a zero-length element would emit
            a non-conformant Type 1 attribute, turning a detectable failure into a
            silent one.
    """
    try:
        element = reference_dataset[keyword]
    except KeyError as error:
        raise KeyError(
            f"Attribute {keyword} is mandatory (Type 1) but is absent from the reference dataset."
        ) from error
    if element.is_empty:
        raise KeyError(
            f"Attribute {keyword} is mandatory (Type 1) but has no value in the reference dataset."
        )
    value = deepcopy(element.value)
    setattr(dataset, keyword, value)


def _copy_type_2(dataset: Dataset, reference_dataset: Dataset, keyword: str) -> None:
    """Copy a Type 2 attribute, which must be present but may be zero-length.

    A reference dataset that omits the attribute is itself non-conformant. The
    attribute is written zero-length rather than omitted, so that the source's
    non-conformance is not propagated into the output.

    Args:
        dataset (Dataset): Target dataset to populate.
        reference_dataset (Dataset): Source dataset to read the attribute from.
        keyword (str): DICOM keyword of the attribute to copy.
    """
    value = deepcopy(reference_dataset[keyword].value) if keyword in reference_dataset else None
    setattr(dataset, keyword, value)


def _copy_type_3(dataset: Dataset, reference_dataset: Dataset, keyword: str) -> None:
    """Copy a Type 3 attribute, which is optional, only when the reference has it.

    Args:
        dataset (Dataset): Target dataset to populate.
        reference_dataset (Dataset): Source dataset to read the attribute from.
        keyword (str): DICOM keyword of the attribute to copy.
    """
    if keyword in reference_dataset:
        setattr(dataset, keyword, deepcopy(reference_dataset[keyword].value))
