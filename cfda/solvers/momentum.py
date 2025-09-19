"""Momentum equation assembler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..core import Mesh, VectorField
from ..core import fv_ops
from ..core.field import ScalarField
from ..core.fv_ops import grad
from ..core.linalg import FvMatrix
from ..numerics.rhie_chow import rhie_chow_face_velocity


@dataclass
class MomentumSystem:
    matrices: list[FvMatrix]
    rhs: np.ndarray
    diag: np.ndarray
    H: np.ndarray
    HbyA: np.ndarray
    rAU: np.ndarray
    rAUf: np.ndarray
    phi: np.ndarray
    face_bc: dict[int, object]


class VectorComponentView:
    def __init__(self, field: VectorField, component: int) -> None:
        self.component = component
        self.mesh = field.mesh
        self.values = field.values[:, component]


class MomentumAssembler:
    def __init__(self, mesh: Mesh, velocity_bcs: Iterable) -> None:
        self.mesh = mesh
        self.velocity_bcs = list(velocity_bcs)
        self._face_bc = {}
        for bc in self.velocity_bcs:
            for fid in bc.faces:
                if fid in self._face_bc:
                    raise ValueError(f"Face {fid} already has a boundary condition assigned")
                self._face_bc[fid] = bc

    def build(
        self,
        velocity: VectorField,
        pressure,
        rho: float,
        mu: float,
        nut,
        alpha_u: float,
    ) -> MomentumSystem:
        mesh = self.mesh
        matrix = FvMatrix(mesh)
        ncells = mesh.ncells
        diag = np.zeros(ncells)
        rhs = np.zeros((ncells, 3))
        gradients = [
            grad(mesh, ScalarField(f"U{comp}", mesh, velocity.values[:, comp]))
            for comp in range(3)
        ]

        convective_rhs = np.zeros((ncells, 3))

        face_velocity = fv_ops.interpolate(mesh, velocity.values)
        for bc in self.velocity_bcs:
            bc.update_face_values(face_velocity, velocity)

        mass_flux = fv_ops.face_flux(mesh, 1.0, face_velocity)
        nut_values = getattr(nut, "values", None)

        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            neigh = face.neighbour
            mf = mass_flux[fid]

            if neigh is None:
                bc = self._face_bc.get(fid)
                phi_face = face_velocity[fid]

                if mf > 0.0:
                    diag[owner] += mf
                else:
                    for comp in range(3):
                        convective_rhs[owner, comp] -= mf * phi_face[comp]

                if bc is not None and bc.is_dirichlet():
                    distance = np.linalg.norm(face.center - mesh.cell_centers[owner])
                    if distance > 0.0:
                        mu_face = mu
                        if nut_values is not None:
                            mu_face += float(nut_values[owner])
                        coeff = mu_face * face.area / distance
                        diag[owner] += coeff
                        for comp in range(3):
                            convective_rhs[owner, comp] += coeff * phi_face[comp]
                continue

            d = mesh.cell_centers[neigh] - mesh.cell_centers[owner]
            distance = np.linalg.norm(d)
            if distance == 0.0:
                continue

            mu_face = mu
            if nut_values is not None:
                mu_face += 0.5 * (nut_values[owner] + nut_values[neigh])
            coeff = mu_face * face.area / distance
            diag[owner] += coeff
            diag[neigh] += coeff
            matrix.add_nb([owner], [neigh], [-coeff])
            matrix.add_nb([neigh], [owner], [-coeff])

            limit_speed = max(
                np.linalg.norm(velocity.values[owner]),
                np.linalg.norm(velocity.values[neigh]),
                1.0,
            )
            flux_limit = limit_speed * face.area
            mf = float(np.clip(mf, -flux_limit, flux_limit))
            mass_flux[fid] = mf

            F_pos = max(mf, 0.0)
            F_neg = min(mf, 0.0)

            diag[owner] += F_pos
            matrix.add_nb([owner], [neigh], [F_neg])

            diag[neigh] += -F_neg
            matrix.add_nb([neigh], [owner], [-F_pos])

            for comp in range(3):
                up = owner if mf >= 0.0 else neigh
                grad_up = gradients[comp][up]
                d_vec = face.center - mesh.cell_centers[up]
                phi_up = velocity.values[up, comp]
                phi_owner = velocity.values[owner, comp]
                phi_neigh = velocity.values[neigh, comp]
                phi_high = phi_up + np.dot(grad_up, d_vec)
                phi_high = float(
                    np.clip(
                        phi_high,
                        min(phi_owner, phi_neigh),
                        max(phi_owner, phi_neigh),
                    )
                )
                delta = mf * (phi_high - phi_up)
                convective_rhs[owner, comp] -= delta
                convective_rhs[neigh, comp] += delta

        diag_relaxed = diag / max(alpha_u, 1e-12)
        matrix.add_diag(range(ncells), diag_relaxed)

        grad_p = grad(mesh, pressure)
        rhs -= grad_p * mesh.cell_volumes[:, None]

        if alpha_u < 1.0:
            factor = (1.0 - alpha_u) / max(alpha_u, 1e-12)
            rhs += factor * diag[:, None] * velocity.values

        rhs += convective_rhs

        component_matrices: list[FvMatrix] = []
        diag_components = np.zeros((ncells, 3))
        H = np.zeros((ncells, 3))

        for comp in range(3):
            comp_matrix = matrix.copy_structure()
            comp_rhs = rhs[:, comp].copy()
            view = VectorComponentView(velocity, comp)
            for bc in self.velocity_bcs:
                bc.apply_coeffs(comp_matrix, comp_rhs, view)
            component_matrices.append(comp_matrix)
            diag_components[:, comp] = comp_matrix._diag.copy()
            H[:, comp] = comp_rhs + diag_components[:, comp] * velocity.values[:, comp]
            rhs[:, comp] = comp_rhs

        diag_effective = diag_components.mean(axis=1)
        rAU = np.zeros_like(diag_effective)
        mask = diag_effective > 1e-30
        rAU[mask] = 1.0 / diag_effective[mask]
        HbyA = np.zeros_like(H)
        HbyA[mask, :] = H[mask, :] * rAU[mask, None]

        rAUf = np.zeros(len(mesh.faces))
        eps = 1e-30
        for fid, face in enumerate(mesh.faces):
            owner = face.owner
            neigh = face.neighbour
            r_owner = max(rAU[owner], eps)
            if neigh is None:
                rAUf[fid] = r_owner
                continue
            r_neigh = max(rAU[neigh], eps)
            denom = (1.0 / r_owner) + (1.0 / r_neigh)
            rAUf[fid] = 0.0 if denom == 0.0 else 2.0 / denom

        face_velocity_star = fv_ops.interpolate(mesh, HbyA)
        for bc in self.velocity_bcs:
            bc.update_face_values(face_velocity_star, velocity)
        phi_star = fv_ops.face_flux(mesh, 1.0, face_velocity_star)

        return MomentumSystem(
            matrices=component_matrices,
            rhs=rhs,
            diag=diag_effective,
            H=H,
            HbyA=HbyA,
            rAU=rAU,
            rAUf=rAUf,
            phi=phi_star,
            face_bc=self._face_bc,
        )

    def face_velocity(self, system: MomentumSystem, velocity: VectorField, pressure) -> np.ndarray:
        return rhie_chow_face_velocity(
            self.mesh, velocity, pressure, system.diag, system.HbyA
        )
