"""Finite-volume diagnostic utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda.core import fv_ops
from cfda.core.bc import ZeroGradient
from cfda.core.field import ScalarField, VectorField
from cfda.core.fv_ops import grad
from cfda.core.linalg import FvMatrix
from cfda.core.mesh import Mesh
from cfda.physics.transport import ConstantTransport
from cfda.solvers.momentum import MomentumAssembler
from cfda.solvers.pressure import PressureAssembler


def geometry_sanity(nx: int = 32, ny: int = 32) -> Dict[str, float]:
    mesh = Mesh.structured(nx, ny)
    volumes = mesh.cell_volumes
    areas = np.array([np.linalg.norm(face.area_vector) for face in mesh.faces])

    # Uniform tangential velocity: [1,0,0]
    U = VectorField("U", mesh, np.tile(np.array([1.0, 0.0, 0.0]), (mesh.ncells, 1)))
    face_flux = fv_ops.face_flux(mesh, 1.0, fv_ops.interpolate(mesh, U.values))

    patch_flux = {}
    for name in mesh.patches():
        flux = [face_flux[fid] for fid in mesh.patch_faces(name)]
        patch_flux[name] = float(np.sum(flux))

    return {
        "volume_min": float(volumes.min()),
        "volume_max": float(volumes.max()),
        "area_min": float(areas.min()),
        "area_max": float(areas.max()),
        "patch_flux": patch_flux,
    }


def _laplacian_matrix(mesh: Mesh, gamma: float = 1.0) -> tuple[FvMatrix, dict[int, tuple[int, float]]]:
    matrix = FvMatrix(mesh)
    diag = np.zeros(mesh.ncells)
    boundary_coeffs: dict[int, tuple[int, float]] = {}
    for fid, face in enumerate(mesh.faces):
        owner = face.owner
        neigh = face.neighbour
        if neigh is None:
            distance = np.linalg.norm(face.center - mesh.cell_centers[owner])
            if distance == 0.0:
                continue
            coeff = gamma * np.linalg.norm(face.area_vector) / distance
            diag[owner] += coeff
            boundary_coeffs[fid] = (owner, coeff)
            continue
        d = mesh.cell_centers[neigh] - mesh.cell_centers[owner]
        distance = np.linalg.norm(d)
        if distance == 0.0:
            continue
        coeff = gamma * np.linalg.norm(face.area_vector) / distance
        diag[owner] += coeff
        diag[neigh] += coeff
        matrix.add_nb([owner], [neigh], [-coeff])
        matrix.add_nb([neigh], [owner], [-coeff])
    matrix.add_diag(range(mesh.ncells), diag)
    return matrix, boundary_coeffs


def laplace_plate(nx: int = 32, ny: int = 32) -> Dict[str, float]:
    mesh = Mesh.structured(nx, ny)
    matrix, boundary_coeffs = _laplacian_matrix(mesh)
    rhs = np.zeros(mesh.ncells)

    aliases = {"top": "ymax", "bottom": "ymin", "left": "xmin", "right": "xmax"}

    def apply_dirichlet(where: str, func):
        patch = aliases.get(where, where)
        faces = mesh.patch_faces(patch)
        for fid in faces:
            if fid not in boundary_coeffs:
                continue
            owner, coeff = boundary_coeffs[fid]
            value = float(func(mesh.faces[fid].center))
            rhs[owner] += coeff * value

    # T = y
    centers = mesh.cell_centers
    T_exact = centers[:, 1]
    apply_dirichlet("top", lambda c: c[1])
    apply_dirichlet("bottom", lambda c: c[1])
    apply_dirichlet("left", lambda c: c[1])
    apply_dirichlet("right", lambda c: c[1])

    T, stats = matrix.solve(rhs=rhs, method="cg", return_stats=True, tol=1e-12)
    return {
        "min": float(T.min()),
        "max": float(T.max()),
        "solver_initial": stats["initial"],
        "solver_final": stats["final"],
        "solver_iterations": stats["iterations"],
        "error_linf": float(np.max(np.abs(T - T_exact))),
    }


def poisson_manufactured(nx: int = 32, ny: int = 32) -> Dict[str, float]:
    mesh = Mesh.structured(nx, ny)
    matrix, boundary_coeffs = _laplacian_matrix(mesh, gamma=1.0)
    rhs = np.zeros(mesh.ncells)

    centers = mesh.cell_centers
    x = centers[:, 0]
    y = centers[:, 1]
    p_exact = np.sin(math.pi * x) * np.sin(math.pi * y)
    source = 2 * (math.pi**2) * p_exact
    rhs += source * mesh.cell_volumes

    aliases = {"top": "ymax", "bottom": "ymin", "left": "xmin", "right": "xmax"}

    def apply_dirichlet(where: str, func):
        patch = aliases.get(where, where)
        faces = mesh.patch_faces(patch)
        for fid in faces:
            if fid not in boundary_coeffs:
                continue
            owner, coeff = boundary_coeffs[fid]
            value = float(func(mesh.faces[fid].center))
            rhs[owner] += coeff * value

    def analytic(point):
        return math.sin(math.pi * point[0]) * math.sin(math.pi * point[1])

    apply_dirichlet("top", analytic)
    apply_dirichlet("bottom", analytic)
    apply_dirichlet("left", analytic)
    apply_dirichlet("right", analytic)

    solution, stats = matrix.solve(rhs=rhs, method="cg", return_stats=True, tol=1e-12)
    error = solution - p_exact
    return {
        "solver_initial": stats["initial"],
        "solver_final": stats["final"],
        "solver_iterations": stats["iterations"],
        "error_l2": float(np.linalg.norm(error) / np.linalg.norm(p_exact)),
        "error_linf": float(np.max(np.abs(error))),
    }


def stokes_cavity(nx: int = 32, ny: int = 32, iterations: int = 60) -> Dict[str, float]:
    # Create a temporary case in memory (structured mesh, lid velocity)
    mesh = Mesh.structured(nx, ny, patch_aliases={"ymax": "bottom", "ymin": "top"})
    U = VectorField("U", mesh, np.zeros((mesh.ncells, 3)))
    p = ScalarField("p", mesh, np.zeros(mesh.ncells))

    # Boundary conditions
    from cfda.core.bc import MovingWall, NoSlipWall

    top_faces = mesh.patch_faces("top")
    bottom_faces = mesh.patch_faces("bottom")
    left_faces = mesh.patch_faces("xmin")
    right_faces = mesh.patch_faces("xmax")

    velocity_bcs = [
        MovingWall("top", mesh, top_faces, [1.0, 0.0, 0.0]),
        NoSlipWall("bottom", mesh, bottom_faces),
        NoSlipWall("xmin", mesh, left_faces),
        NoSlipWall("xmax", mesh, right_faces),
    ]

    momentum = MomentumAssembler(mesh, velocity_bcs)
    transport = ConstantTransport(rho=1.0, mu=1.0)

    pressure_bc = [
        ZeroGradient("top", mesh, top_faces),
        ZeroGradient("bottom", mesh, bottom_faces),
        ZeroGradient("xmin", mesh, left_faces),
        ZeroGradient("xmax", mesh, right_faces),
    ]
    reference = (0, 0.0)
    pressure_assembler = PressureAssembler(mesh, pressure_bc, reference)

    cumulative_mass = 0.0
    stats = []
    for outer in range(1, iterations + 1):
        mom_sys = momentum.build(
            velocity=U,
            pressure=p,
            rho=transport.density(),
            mu=transport.viscosity(),
            nut=None,
            alpha_u=0.3,
        )

        for comp in range(3):
            prev = U.values[:, comp].copy()
            sol, lin_stats = mom_sys.matrices[comp].solve(
                mom_sys.rhs[:, comp],
                method="bicgstab",
                return_stats=True,
                tol=1e-8,
                initial_guess=prev,
            )
            U.values[:, comp] = sol

        rho = transport.density()
        phi = mom_sys.phi.copy()
        p_initial = 0.0
        p_final = 0.0
        p_rel = 0.0
        n_corr = 2
        beta_phi = 1.0
        for corr in range(n_corr):
            alpha_p = 1.0
            pressure_system = pressure_assembler.build(
                phi,
                pressure=p,
                rho=rho,
                alpha_p=alpha_p,
                rAUf=mom_sys.rAUf,
            )
            p_corr, p_stats = pressure_system.matrix.solve(
                pressure_system.rhs,
                method="bicgstab",
                return_stats=True,
                tol=1e-8,
            )
            p.values += alpha_p * p_corr
            p_initial = max(p_initial, p_stats["initial"])
            p_final = max(p_final, p_stats["final"])
            p_rel = max(p_rel, p_stats["relative"])
            grad_pcorr = grad(
                mesh,
                p_corr,
                pressure_assembler.pressure_bcs,
            )
            for comp in range(3):
                U.values[:, comp] -= mom_sys.rAU * grad_pcorr[:, comp]

            centers = mesh.cell_centers
            phi_prev = phi.copy()
            delta_phi = np.zeros_like(phi)
            for fid, face in enumerate(mesh.faces):
                neigh = face.neighbour
                if neigh is None:
                    continue
                owner = face.owner
                vec = centers[neigh] - centers[owner]
                distance = float(np.linalg.norm(vec))
                if distance <= 0.0:
                    continue
                n_hat = vec / distance
                s_proj = float(np.dot(face.area_vector, n_hat))
                dp = float(p_corr[neigh] - p_corr[owner])
                dpdn = dp * (s_proj / distance)
                correction = beta_phi * mom_sys.rAUf[fid] * dpdn
                phi[fid] -= correction
                delta_phi[fid] = correction

            phi_expected = phi_prev - delta_phi
            if not np.allclose(phi, phi_expected, rtol=1e-9, atol=1e-12):
                diff = np.abs(phi - phi_expected)
                raise AssertionError(
                    f"Momentum probe flux update mismatch max diff {diff.max():.3e}"
                )

            assert np.isfinite(phi).all()
            assert np.isfinite(U.values).all()
            assert np.isfinite(p.values).all()

            div_phi_corr = fv_ops.div(mesh, phi)
            mass_corr = float(np.sqrt(np.sum(mesh.cell_volumes * div_phi_corr**2)))
            print(
                f"    corrector {corr + 1}: ||div(phi)||_V = {mass_corr:.6e}"
            )

        div_phi = fv_ops.div(mesh, phi)
        mass_norm = float(np.sqrt(np.sum(mesh.cell_volumes * div_phi**2)))
        global_mass = float(np.sum(phi))
        cumulative_mass += global_mass
        stats.append((outer, mass_norm, global_mass, cumulative_mass))
        if mass_norm < 1e-6:
            break

    return {
        "iterations": len(stats),
        "mass_history": stats,
    }


def main():
    print("-- Geometry sanity --")
    geom = geometry_sanity()
    print(geom)

    print("\n-- Laplace plate --")
    lap = laplace_plate()
    print(lap)

    print("\n-- Poisson manufactured --")
    poi = poisson_manufactured()
    print(poi)

    print("\n-- Stokes cavity (no convection) --")
    stokes = stokes_cavity()
    print(stokes)


if __name__ == "__main__":
    main()
