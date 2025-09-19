"""Pressure correction assembler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..core import Mesh, ScalarField
from ..core import fv_ops
from ..core.fv_ops import grad
from ..core.linalg import FvMatrix


@dataclass
class PressureSystem:
    matrix: FvMatrix
    rhs: np.ndarray
    mass_flux: np.ndarray
    rAUf: np.ndarray


class PressureAssembler:
    def __init__(
        self,
        mesh: Mesh,
        pressure_bcs: Iterable,
        reference: tuple[int, float] | None = None,
    ) -> None:
        self.mesh = mesh
        self.pressure_bcs = list(pressure_bcs)
        self.reference = reference
        self._face_bc = {}
        for bc in self.pressure_bcs:
            for fid in bc.faces:
                if fid in self._face_bc:
                    raise ValueError(f"Face {fid} already has a boundary condition assigned")
                self._face_bc[fid] = bc

    def build(
        self,
        face_flux: np.ndarray,
        pressure: ScalarField,
        rho: float,
        alpha_p: float,
        rAUf: np.ndarray | None = None,
    ) -> PressureSystem:
        mesh = self.mesh
        div_flux = fv_ops.div(mesh, face_flux)
        rhs = -div_flux * mesh.cell_volumes

        matrix = FvMatrix(mesh)
        diag = np.zeros(mesh.ncells)
        face_pressure = fv_ops.interpolate(mesh, pressure.values)
        for bc in self.pressure_bcs:
            bc.update_face_values(face_pressure, pressure)

        grad_p = grad(mesh, pressure, self.pressure_bcs)

        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            neigh = face.neighbour
            coeff_scale = rAUf[fid] if rAUf is not None else 1.0

            if neigh is None:
                bc = self._face_bc.get(fid)
                if bc is not None and bc.is_dirichlet():
                    vec = face.center - mesh.cell_centers[owner]
                    distance = float(np.linalg.norm(vec))
                    if distance > 0.0:
                        n_hat = vec / distance
                        projection = float(np.dot(face.area_vector, n_hat))
                        coeff = coeff_scale * projection / distance
                        diag[owner] += coeff
                        rhs[owner] += coeff * face_pressure[fid]
                continue

            d = mesh.cell_centers[neigh] - mesh.cell_centers[owner]
            distance = float(np.linalg.norm(d))
            if distance == 0.0:
                continue
            n_hat = d / distance
            projection = float(np.dot(face.area_vector, n_hat))
            coeff = coeff_scale * projection / max(distance, 1e-30)
            diag[owner] += coeff
            diag[neigh] += coeff
            matrix.add_nb([owner], [neigh], [-coeff])
            matrix.add_nb([neigh], [owner], [-coeff])

            # Non-orthogonal correction source term (vanishes on orthogonal meshes)
            non_orth_vec = face.area_vector - n_hat * projection
            if np.linalg.norm(non_orth_vec) > 0.0:
                grad_face = 0.5 * (grad_p[owner] + grad_p[neigh])
                correction = coeff_scale * float(np.dot(non_orth_vec, grad_face))
                rhs[owner] -= correction
                rhs[neigh] += correction
        matrix.add_diag(range(mesh.ncells), diag)

        for bc in self.pressure_bcs:
            bc.apply_coeffs(matrix, rhs, pressure)

        if self.reference is not None:
            self._apply_reference(matrix, rhs, self.reference[0], self.reference[1])

        return PressureSystem(matrix=matrix, rhs=rhs, mass_flux=face_flux, rAUf=rAUf if rAUf is not None else np.ones(len(mesh.faces)))

    def _apply_reference(self, matrix: FvMatrix, rhs: np.ndarray, cell: int, value: float) -> None:
        # Zero contributions from cell row
        keys_to_remove = [key for key in matrix._offdiag if key[0] == cell]
        for key in keys_to_remove:
            del matrix._offdiag[key]

        # Adjust other rows that couple to reference cell
        for (row, col), coeff in list(matrix._offdiag.items()):
            if col == cell:
                rhs[row] -= coeff * value
                del matrix._offdiag[(row, col)]

        matrix._diag[cell] = 1.0
        rhs[cell] = value
