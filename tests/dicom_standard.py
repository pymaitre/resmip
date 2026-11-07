"""Edition and location of the DICOM standard used by ``dicom-validator`` in tests.

The standard is stored next to this module, not in the user's home directory, so that
the test fixture and the CI cache step resolve the same directory by construction. The
path is anchored on ``__file__`` so that it does not depend on the operating system's
home directory or on the working directory pytest is started from.

Run as ``python -m tests.dicom_standard`` to print the cache path and key as
``name=value`` lines for ``$GITHUB_OUTPUT``.
"""

from pathlib import Path

import dicom_validator

DICOM_EDITION = "2026d"
"""Edition of the DICOM standard that written files are validated against.

NEMA publishes several editions a year (``a`` to ``e``). Each edition includes all earlier
changes, so there is no need to follow every release. Review this value once a year, and
sooner when resmip starts writing an attribute, module or IOD introduced or changed by a
later Supplement or Correction Proposal. A new edition can change the outcome of the
``*_is_dicom_conformant`` tests: treat new findings as defects to investigate, not as a
reason to revert the edition.
"""
DICOM_STANDARD_PATH = Path(__file__).resolve().parent / ".dicom-standard"


def cache_key(edition: str, validator_version: str) -> str:
    """Return the CI cache key for the DICOM standard.

    The key must change whenever the cached content would be stale: a different edition
    needs different DocBook files, and a different ``dicom-validator`` version rebuilds the
    JSON derived from them, because ``EditionReader`` compares the version stored in the
    cache with its own exactly. The Python version and the operating system are not part
    of this key; the workflow adds the operating system itself.

    Args:
        edition: DICOM edition identifier, e.g. ``"2026d"``.
        validator_version: Version of the installed ``dicom-validator`` package.

    Returns:
        A key that identifies the cached standard for this edition and validator version.
    """
    return f"{edition}-{validator_version}"


def cache_path() -> str:
    """Return the DICOM standard directory relative to the repository root.

    ``actions/cache`` resolves relative paths against the workspace, which is the
    repository root, and the path is printed into ``$GITHUB_OUTPUT``. It is therefore
    derived from ``__file__`` rather than from the working directory, and written with
    forward slashes so that it is identical on every runner operating system.

    Returns:
        The POSIX-style path of ``DICOM_STANDARD_PATH`` relative to the repository root.
    """
    repository_root = Path(__file__).resolve().parents[1]
    return DICOM_STANDARD_PATH.relative_to(repository_root).as_posix()


if __name__ == "__main__":
    print(f"path={cache_path()}")
    print(f"key={cache_key(DICOM_EDITION, dicom_validator.__version__)}")
