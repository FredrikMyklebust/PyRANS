"""Finite-volume helper operations."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .field import Field
from .mesh import Mesh


def _values_array(field: Field | np.ndarray | Iterable[float]) -> np.ndarray:
    if isinstance(field, Field):
        return field.values
    return np.asarray(field, dtype=float)


def interpolate(
    mesh: Mesh,
    field: Field | np.ndarray,
    scheme: str = "linear",
    face_flux: Optional[np.ndarray] = None,
) -> np.ndarray:
    values = _values_array(field)
    scalar = values.ndim == 1
    nfaces = len(mesh.faces)
    if scalar:
        face_vals = np.zeros(nfaces)
    else:
        face_vals = np.zeros((nfaces, values.shape[1]))

    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        neighbour = face.neighbour
        if neighbour is None:
            face_vals[fid] = values[owner]
            continue
        if scheme.lower() == "upwind":
            if face_flux is None:
                raise ValueError("Upwind interpolation requires face_flux")
            src = owner if face_flux[fid] >= 0.0 else neighbour
            face_vals[fid] = values[src]
        else:
            face_vals[fid] = 0.5 * (values[owner] + values[neighbour])
    return face_vals


def grad(
    mesh: Mesh,
    field: Field | np.ndarray,
    bcs: Optional[Iterable] = None,
) -> np.ndarray:
    values = _values_array(field)
    ncomp = 1 if values.ndim == 1 else values.shape[1]
    grads = np.zeros((mesh.ncells, 3, ncomp)) if ncomp > 1 else np.zeros((mesh.ncells, 3))

    face_vals = interpolate(mesh, values)
    if bcs is not None:
        for bc in bcs:
            bc.update_face_values(face_vals, field)

    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        phi_f = face_vals[fid]
        Sf = face.area_vector
        if ncomp == 1:
            grads[owner] += phi_f * Sf
        else:
            grads[owner] += np.outer(Sf, phi_f)

        neighbour = face.neighbour
        if neighbour is None:
            continue
        if ncomp == 1:
            grads[neighbour] -= phi_f * Sf
        else:
            grads[neighbour] -= np.outer(Sf, phi_f)

    vols = mesh.cell_volumes
    if ncomp == 1:
        return grads / vols[:, None]
    return grads / vols[:, None, None]


def div(mesh: Mesh, face_flux: np.ndarray) -> np.ndarray:
    if face_flux.ndim != 1:
        raise ValueError("div expects scalar flux per face")
    divergence = np.zeros(mesh.ncells)
    for fid, face in enumerate(mesh.faces):
        flux = face_flux[fid]
        divergence[face.owner] += flux
        if face.neighbour is not None:
            divergence[face.neighbour] -= flux
    return divergence / mesh.cell_volumes


def laplacian(mesh: Mesh, gamma: Field | np.ndarray | float, field: Field | np.ndarray) -> np.ndarray:
    gamma_values = _values_array(gamma) if not isinstance(gamma, (float, int)) else gamma
    phi = _values_array(field)
    if phi.ndim != 1:
        raise ValueError("laplacian currently implemented for scalar fields only")
    diffusion = np.zeros(mesh.ncells)
    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        neighbour = face.neighbour
        if isinstance(gamma_values, np.ndarray):
            g_face = gamma_values[owner]
            if neighbour is not None:
                g_face = 0.5 * (gamma_values[owner] + gamma_values[neighbour])
        else:
            g_face = float(gamma_values)
        if neighbour is None:
            continue
        delta = phi[neighbour] - phi[owner]
        d = np.linalg.norm(mesh.cell_centers[neighbour] - mesh.cell_centers[owner])
        coeff = g_face * face.area / d
        diffusion[owner] += coeff * delta
        diffusion[neighbour] -= coeff * delta
    return diffusion / mesh.cell_volumes


def face_flux(
    mesh: Mesh,
    density: float | np.ndarray,
    face_velocity: np.ndarray,
) -> np.ndarray:
    if isinstance(density, (float, int)):
        rho = float(density)
    else:
        rho_cell = _values_array(density)
        rho = np.zeros(len(mesh.faces))
    flux = np.zeros(len(mesh.faces))
    for fid, face in enumerate(mesh.faces):
        vf = face_velocity[fid]
        if isinstance(rho, float):
            rho_f = rho
        else:
            owner = face.owner
            neighbour = face.neighbour
            if neighbour is None:
                rho_f = rho_cell[owner]
            else:
                rho_f = 0.5 * (rho_cell[owner] + rho_cell[neighbour])
            rho[fid] = rho_f
        flux[fid] = rho_f * np.dot(vf, face.area_vector)
    return flux
