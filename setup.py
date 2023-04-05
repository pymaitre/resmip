"""
    Installation script for srmip.
"""

from pathlib import Path
import re
from setuptools import setup, find_packages

package_name = "srmip"
package_root = "src"
repository_root = Path(__file__).parent
requirements = (repository_root / "Requirements.txt").read_text()

description = "Python library for Medical Image Processing (MIP)."
long_description = (repository_root / "README.md").read_text()


def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    version_file = repository_root / f"{package_root}/{package_name}/__init__.py"
    initfile_lines = version_file.open("rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    return "unknown"


setup(
    name=package_name,
    version=get_version(),
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="None",
    author="G. Palazzo, M.G. Ubeira-Gabellini",
    author_email="",
    url="https://github.com/SRresearch/srmip",
    package_dir={"": package_root},
    packages=find_packages(package_root),
    zip_safe=False,
    classifiers=[],
    package_data={
        "": ["*.yaml"],
    },
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={"test": ["prospector", "pytest"]},
)