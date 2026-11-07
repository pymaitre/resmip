"""Edition and location of the DICOM standard used by ``dicom-validator`` in tests.

The standard is stored next to this module, not in the user's home directory, so that
the test fixture and the CI cache step resolve the same directory by construction. The
path is anchored on ``__file__`` so that it does not depend on the operating system's
home directory or on the working directory pytest is started from.
"""

from pathlib import Path

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
