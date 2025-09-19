"""Rhie-Chow interpolation utilities."""

from __future__ import annotations

import numpy as np

from ..core.field import ScalarField, VectorField
from ..core.mesh import Mesh


def rhie_chow_face_velocity(
    mesh: Mesh,
    velocity: VectorField,
    pressure: ScalarField,
    a_diag: np.ndarray,
    H_over_a: np.ndarray,
) -> np.ndarray:
    """Compute Rhie–Chow corrected face velocity.

    The correction follows the classical collocated formulation where the
    interpolated pseudo-velocity H/a replaces the naive linear interpolation and
    a pressure-gradient contribution is subtracted using the momentum diagonal
    coefficients to prevent checkerboarding.
    """

    nfaces = len(mesh.faces)
    face_velocity = np.zeros((nfaces, 3))
    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        neigh = face.neighbour

        if neigh is None:
            # Boundary: use local velocity, BCs already enforce desired value
            face_velocity[fid] = velocity.values[owner]
            continue

        u_lin = 0.5 * (velocity.values[owner] + velocity.values[neigh])
        u_hat = 0.5 * (H_over_a[owner] + H_over_a[neigh])
        delta_hat = u_hat - u_lin
        limit = max(np.linalg.norm(u_lin), 1.0)
        delta_norm = np.linalg.norm(delta_hat)
        if delta_norm > 5.0 * limit:
            delta_hat *= (5.0 * limit) / (delta_norm + 1e-12)
        u_hat = u_lin + delta_hat

        delta_p = pressure.values[neigh] - pressure.values[owner]
        d_vec = mesh.cell_centers[neigh] - mesh.cell_centers[owner]
        d_normal = float(np.dot(d_vec, face.normal))
        if abs(d_normal) < 1e-12:
            d_normal = np.linalg.norm(d_vec)
        grad_p = delta_p / max(d_normal, 1e-12)

        a_face = 0.5 * (a_diag[owner] + a_diag[neigh]) if a_diag is not None else 1.0
        correction = -(grad_p / max(a_face, 1e-12)) * face.normal

        face_velocity[fid] = u_hat + correction

    return face_velocity
