"""Outlet boundary conditions."""

from __future__ import annotations

from ..field import ScalarField
from ..linalg import FvMatrix
from .base import BoundaryCondition


class PressureOutlet(BoundaryCondition):
    def __init__(self, name, mesh, faces, value: float = 0.0):
        super().__init__(name, mesh, faces)
        self.pressure = float(value)

    def apply_coeffs(self, matrix: FvMatrix, rhs, field) -> None:
        return

    def update_face_values(self, face_values, field) -> None:
        for fid in self.faces:
            face_values[fid] = self.pressure

    def is_dirichlet(self) -> bool:
        return True
