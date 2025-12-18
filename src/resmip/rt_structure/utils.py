"""Utility functions for RTStructures and RTStructureSets."""

from pathlib import Path


def get_structure_name_from_filename(filename: Path) -> str:
    """Get the structure name from the filename.

    If the file is compressed, e.g.: structure.nii.gz, remove ".nii".

    Args:
        filename (Path): Name of the file.

    Returns:
        str: Name of the RT Structure.
    """
    compress_extensions = [".gz"]
    if filename.suffix in compress_extensions:
        return ".".join(filename.stem.split(".")[:-1])
    return filename.stem
