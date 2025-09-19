"""Zero-gradient (Neumann) boundary condition."""

from __future__ import annotations

from .base import BoundaryCondition


class ZeroGradient(BoundaryCondition):
    def apply_coeffs(self, matrix, rhs, field) -> None:  # pragma: no cover - no-op
        return

    def update_face_values(self, face_values, field) -> None:
        values = getattr(field, "values", field)
        for fid in self.faces:
            owner = self.mesh.faces[fid].owner
            face_values[fid] = values[owner]
