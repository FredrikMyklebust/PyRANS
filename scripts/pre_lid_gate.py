"""Pre-lid diagnostic gate for mass-conservation sanity checks."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import fv_diagnostics
from cfda.run.case import Case
from cfda.core import fv_ops
from cfda.core.fv_ops import grad
from cfda.core.mesh import Mesh


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


TOL_GEOM = 1e-12


def geometry_gate() -> GateResult:
    stats = fv_diagnostics.geometry_sanity()
    dx = 1.0 / 32
    expected_volume = dx * dx
    expected_area = dx
    ok = (
        abs(stats["volume_min"] - expected_volume) < 1e-14
        and abs(stats["volume_max"] - expected_volume) < 1e-14
        and abs(stats["area_min"] - expected_area) < 1e-14
        and abs(stats["area_max"] - expected_area) < 1e-14
        and abs(stats["patch_flux"]["xmin"] - 1.0) < 1e-12
        and abs(stats["patch_flux"]["xmax"] + 1.0) < 1e-12
        and abs(stats["patch_flux"]["ymin"]) < 1e-12
        and abs(stats["patch_flux"]["ymax"]) < 1e-12
    )
    detail = (
        f"vol=[{stats['volume_min']:.3e},{stats['volume_max']:.3e}] "
        f"area=[{stats['area_min']:.3e},{stats['area_max']:.3e}]"
    )
    return GateResult("geometry", ok, detail)


def laplace_gate() -> GateResult:
    stats = fv_diagnostics.laplace_plate()
    ok = stats["error_linf"] < 1e-10
    return GateResult("laplace_dirichlet", ok, f"Linf={stats['error_linf']:.2e}")


def poisson_gate() -> GateResult:
    stats32 = fv_diagnostics.poisson_manufactured(32, 32)
    stats64 = fv_diagnostics.poisson_manufactured(64, 64)
    ratio = stats32["error_l2"] / stats64["error_l2"] if stats64["error_l2"] > 0 else math.inf
    ok = stats32["error_l2"] < 1e-3 and ratio > 3.0
    detail = f"L2_32={stats32['error_l2']:.2e}, L2_64={stats64['error_l2']:.2e}, ratio={ratio:.2f}"
    return GateResult("poisson_manufactured", ok, detail)


def discrete_gauss_gate() -> GateResult:
    mesh = Mesh.structured(32, 32)
    centers = mesh.cell_centers
    phi = centers[:, 0] ** 2 + centers[:, 1] ** 2
    face_phi = fv_ops.interpolate(mesh, phi)
    face_vel = np.zeros((len(mesh.faces), 3))
    face_vel[:, 0] = face_phi
    phi_flux = fv_ops.face_flux(mesh, 1.0, face_vel)
    boundary_sum = sum(phi_flux[fid] for fid, face in enumerate(mesh.faces) if face.patch)
    volume_sum = np.sum(fv_ops.div(mesh, phi_flux) * mesh.cell_volumes)
    diff = abs(boundary_sum - volume_sum)
    return GateResult("discrete_gauss", diff < 1e-12, f"diff={diff:.2e}")


def stokes_gate() -> GateResult:
    stats = fv_diagnostics.stokes_cavity(nx=16, ny=16, iterations=10)
    history = stats["mass_history"]
    first = history[0][1]
    last = history[-1][1]
    ok = last < first * 0.5 or last < 1e-6
    detail = f"mass0={first:.2e}, massN={last:.2e}"
    return GateResult("stokes_mass_drop", ok, detail)


def rhs_sign_gate() -> GateResult:
    case = Case.from_yaml("tests/cases/simple/system/case.yaml")
    mesh = case.mesh
    momentum = case.momentum_assembler.build(
        velocity=case.U,
        pressure=case.p,
        rho=case.transport.density(),
        mu=case.transport.viscosity(),
        nut=case.turbulence_model.nut(),
        alpha_u=case.coupling.alpha_u,
    )
    phi_star = momentum.phi.copy()
    base = np.sqrt(np.sum(mesh.cell_volumes * fv_ops.div(mesh, phi_star) ** 2))
    system = case.pressure_assembler.build(
        phi_star,
        pressure=case.p,
        rho=case.transport.density(),
        alpha_p=1.0,
        rAUf=momentum.rAUf,
    )
    p_corr = system.matrix.solve(system.rhs, method=case.coupling.solver_p, tol=1e-10)
    phi = phi_star.copy()
    centers = mesh.cell_centers
    for fid, face in enumerate(mesh.faces):
        neigh = face.neighbour
        if neigh is None:
            continue
        owner = face.owner
        vec = centers[neigh] - centers[owner]
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-12:
            continue
        n_hat = vec / dist
        s_proj = float(np.dot(face.area_vector, n_hat))
        dpdn = float(p_corr[neigh] - p_corr[owner]) * (s_proj / dist)
        phi[fid] -= momentum.rAUf[fid] * dpdn
    after = np.sqrt(np.sum(mesh.cell_volumes * fv_ops.div(mesh, phi) ** 2))
    ok = after < base * 0.5
    detail = f"mass_before={base:.2e}, mass_after={after:.2e}"
    return GateResult("rhs_sign", ok, detail)


def run_gate() -> int:
    checks = [
        geometry_gate(),
        discrete_gauss_gate(),
        laplace_gate(),
        poisson_gate(),
        stokes_gate(),
        rhs_sign_gate(),
    ]
    failures = [c for c in checks if not c.passed]
    print("Pre-lid diagnostic gate")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name:20s} {check.detail}")
    if failures:
        print(f"\n{len(failures)} gate(s) failed. Fix before attempting the lid-driven cavity.")
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_gate())
