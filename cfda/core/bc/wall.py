"""Wall boundary conditions."""

from __future__ import annotations

import numpy as np

from .base import BoundaryCondition
from ..linalg import FvMatrix


class NoSlipWall(BoundaryCondition):
    def __init__(self, name, mesh, faces):
        super().__init__(name, mesh, faces)
        self.velocity = np.zeros(3)

    def apply_coeffs(self, matrix: FvMatrix, rhs: np.ndarray, field) -> None:
        return

    def is_dirichlet(self) -> bool:
        return True

    def update_face_values(self, face_values: np.ndarray, field) -> None:
        component = getattr(field, "component", None)
        for fid in self.faces:
            if face_values.ndim == 2:
                face_values[fid, :] = self.velocity
            elif component is not None:
                face_values[fid] = self.velocity[component]
            else:
                face_values[fid] = 0.0


class MovingWall(NoSlipWall):
    def __init__(self, name, mesh, faces, velocity):
        super().__init__(name, mesh, faces)
        self.velocity = np.asarray(velocity, dtype=float)
