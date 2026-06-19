"""Rotation-matrix builders and direction-cosine helpers."""

import numpy as np


def _rx(angle: float) -> np.ndarray:
    """Build the rotation matrix for a right-handed rotation about the X axis.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 active rotation matrix.

    """
    return np.array(
        (
            (1, 0, 0),
            (0, np.cos(angle), -np.sin(angle)),
            (0, np.sin(angle), np.cos(angle)),
        )
    )


def _ry(angle: float) -> np.ndarray:
    """Build the rotation matrix for a right-handed rotation about the Y axis.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 active rotation matrix.

    """
    return np.array(
        (
            (np.cos(angle), 0, np.sin(angle)),
            (0, 1, 0),
            (-np.sin(angle), 0, np.cos(angle)),
        )
    )


def _rz(angle: float) -> np.ndarray:
    """Build the rotation matrix for a right-handed rotation about the Z axis.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 active rotation matrix.

    """
    return np.array(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
    )


def _nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    """Project a matrix onto the nearest orthogonal matrix via SVD.

    Computes the orthogonal Procrustes solution ``U @ Vt`` from the singular
    value decomposition, i.e. the orthogonal matrix closest to ``matrix`` in
    the Frobenius norm. The determinant sign is preserved, so a near-reflection
    is projected to a reflection rather than being forced to a proper rotation.

    Args:
        matrix (np.ndarray): 3x3 matrix to project, expected to be close to
            orthogonal (e.g. a direction cosine matrix with rounding error).

    Returns:
        np.ndarray: Nearest 3x3 orthogonal matrix, with the same determinant
            sign as ``matrix``.

    """
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def _euler_zxy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation matrix into intrinsic Z-X-Y Euler angles.

    Recovers the angles such that ``_rz(angle_z) @ _rx(angle_x) @ _ry(angle_y)``
    reconstructs ``rotation``. The input is assumed to be a proper rotation
    (orthogonal, determinant +1); callers are responsible for validating it.

    At the gimbal-lock singularity (``angle_x = +/- pi/2``, where
    ``cos(angle_x) ~ 0``) ``angle_y`` and ``angle_z`` become degenerate and only
    their combination is observable. In that case ``angle_y`` is pinned to 0 and
    the remaining degree of freedom is recovered into ``angle_z``, so the
    decomposition stays exact over the whole rotation group.

    Args:
        rotation (np.ndarray): 3x3 proper rotation matrix.

    Returns:
        tuple[float, float, float]: Rotation angles ``(angle_x, angle_y,
            angle_z)`` in radians, in the Z-X-Y convention.

    """
    cos_x = np.sqrt(rotation[2, 0] ** 2 + rotation[2, 2] ** 2)
    angle_x = np.arctan2(rotation[2, 1], cos_x)
    if np.isclose(cos_x, 0.0, atol=1e-7):
        angle_y = 0
        angle_z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        angle_y = np.arctan2(-rotation[2, 0], rotation[2, 2])
        angle_z = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return angle_x, angle_y, angle_z
