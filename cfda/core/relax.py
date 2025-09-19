"""Under-relaxation helpers."""

from __future__ import annotations

import numpy as np


def implicit_relaxation(diagonal: np.ndarray, source: np.ndarray, field_values: np.ndarray, alpha: float):
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if alpha == 1.0:
        return diagonal, source
    diag_relaxed = diagonal / alpha
    correction = (1.0 - alpha) / alpha * diagonal * field_values
    return diag_relaxed, source + correction
