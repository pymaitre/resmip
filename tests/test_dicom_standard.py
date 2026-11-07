"""Tests for the DICOM standard cache location and key used by CI."""

from .dicom_standard import cache_key, cache_path


def test_cache_key_changes_when_validator_version_changes():
    """A dicom-validator upgrade must not restore JSON built by the previous version."""
    edition = "2026d"

    key_before_upgrade = cache_key(edition, "0.8.2")
    key_after_upgrade = cache_key(edition, "0.9.0")

    assert key_before_upgrade != key_after_upgrade


def test_cache_key_changes_when_edition_changes():
    """An edition bump must not restore the DocBook files of the previous edition."""
    validator_version = "0.8.2"

    key_before_bump = cache_key("2026c", validator_version)
    key_after_bump = cache_key("2026d", validator_version)

    assert key_before_bump != key_after_bump


def test_cache_path_is_posix_path_relative_to_repository_root():
    """The workflow resolves the path against the workspace on every runner OS."""
    expected_path = "tests/.dicom-standard"

    path = cache_path()

    assert path == expected_path
