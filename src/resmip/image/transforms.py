"""Transforms."""

import numpy as np


def _rx(angle):
    """"""
    return np.array(
        (
            (1, 0, 0),
            (0, np.cos(angle), -np.sin(angle)),
            (0, np.sin(angle), np.cos(angle)),
        )
    )


def _ry(angle):
    """"""
    return np.array(
        (
            (np.cos(angle), 0, np.sin(angle)),
            (0, 1, 0),
            (-np.sin(angle), 0, np.cos(angle)),
        )
    )


def _rz(angle):
    """"""
    return np.array(
        (
            (np.cos(angle), -np.sin(angle), 0),
            (np.sin(angle), np.cos(angle), 0),
            (0, 0, 1),
        )
    )
