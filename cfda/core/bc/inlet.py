"""Inlet boundary conditions."""

from __future__ import annotations

import numpy as np

from .base import BoundaryCondition
from ..linalg import FvMatrix


class VelocityInlet(BoundaryCondition):
    def __init__(self, name, mesh, faces, value):
        super().__init__(name, mesh, faces)
        self.velocity = np.asarray(value, dtype=float)

    def apply_coeffs(self, matrix: FvMatrix, rhs: np.ndarray, field) -> None:
        return

    def is_dirichlet(self) -> bool:
        return True

    def update_face_values(self, face_values: np.ndarray, field) -> None:
        for fid in self.faces:
            if face_values.ndim == 2:
                face_values[fid, :] = self.velocity
            else:
                component = getattr(field, "component", None)
                value = self.velocity[component] if component is not None else 0.0
                face_values[fid] = value
