"""Validation runner for lid-driven cavity and channel benchmarks.

This script executes three canonical cases:

- Lid-driven cavity (Re=100 by default) using the existing benchmark
  harness.
- Plane Poiseuille flow driven by a constant pressure gradient.
- Plane Couette flow driven by a moving lid.

For each case it records basic error metrics and writes profile plots to the
`tests/artifacts` directory (the same location used by the test suite).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda import Case
from cfda.core.bc import MovingWall, NoSlipWall
from cfda.core.field import ScalarField, VectorField
from cfda.core.mesh import Mesh
from cfda.physics.transport import ConstantTransport
from cfda.physics.turbulence.laminar import LaminarModel
from cfda.solvers.momentum import MomentumAssembler

from scripts.lid_benchmark import run_simulation as run_lid_simulation

ARTIFACT_DIR = Path("tests/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def structured_mesh(nx: int, ny: int) -> Mesh:
    return Mesh.structured(
        nx,
        ny,
        lengths=(1.0, 1.0),
        patch_aliases={"ymin": "top", "ymax": "bottom", "xmin": "inlet", "xmax": "outlet"},
    )


def solve_momentum(mesh: Mesh, pressure: ScalarField, velocity_bcs) -> VectorField:
    transport = ConstantTransport(rho=1.0, mu=0.01)
    model = LaminarModel(mesh, {}, transport, {})
    velocity = VectorField("U", mesh, np.zeros((mesh.ncells, 3)))
    assembler = MomentumAssembler(mesh, velocity_bcs)
    system = assembler.build(
        velocity=velocity,
        pressure=pressure,
        rho=transport.density(),
        mu=transport.viscosity(),
        nut=model.nut(),
        alpha_u=1.0,
    )
    for comp in range(3):
        velocity.values[:, comp] = system.matrices[comp].solve(system.rhs[:, comp])
    return velocity


def average_by_height(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape(ny, nx).mean(axis=1)


def plot_profile(filename: str, y: np.ndarray, numeric: np.ndarray, analytic: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(numeric, y, label="CFDA")
    ax.plot(analytic, y, "--", label="Analytic")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.savefig(ARTIFACT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ValidationResult:
    name: str
    metrics: Dict[str, float]


def run_lid(re: int, mesh: int) -> ValidationResult:
    result = run_lid_simulation(
        Re=re,
        nx=mesh,
        alpha_u=0.25,
        alpha_p=0.3,
        min_outer=60,
        max_outer=120,
        tol_u=1e-7,
        tol_p=1e-7,
        tol_mass=1e-5,
    )
    metrics = {
        "iterations": result.iterations,
        "final_U_res": result.final_U_res,
        "final_mass_res": result.final_mass_res,
        "u_L2": result.u_L2,
        "v_L2": result.v_L2,
        "vortex_x": result.vortex_center[0],
        "vortex_y": result.vortex_center[1],
        "vortex_psi": result.vortex_streamfunction,
    }
    return ValidationResult(name=f"lid_Re{re}_{mesh}", metrics=metrics)


def run_poiseuille(nx: int = 40, ny: int = 80, G: float = 1.0) -> ValidationResult:
    mesh = structured_mesh(nx, ny)
    centers_x = mesh.cell_centers[:, 0]
    pressure = ScalarField("p", mesh, -G * centers_x)
    bcs = [
        NoSlipWall("bottom", mesh, mesh.patch_faces("bottom")),
        NoSlipWall("top", mesh, mesh.patch_faces("top")),
    ]
    velocity = solve_momentum(mesh, pressure, bcs)
    ux = velocity.values[:, 0]
    profile = average_by_height(ux, nx, ny)
    H = mesh.spacing[1] * ny
    y = (np.arange(ny) + 0.5) * (H / ny)
    mu = 0.01
    analytic = (G / (2 * mu)) * y * (H - y)
    bulk_numeric = ux.mean()
    dpdx = -G
    bulk_expected = (-dpdx) * H**2 / (12 * mu)
    rel_err = float(np.linalg.norm(profile - analytic) / np.linalg.norm(analytic))
    bulk_err = float(abs(bulk_numeric - bulk_expected) / abs(bulk_expected))
    plot_profile("poiseuille_profile.png", y, profile, analytic, "Plane Poiseuille")
    return ValidationResult(
        name="poiseuille",
        metrics={
            "rel_profile_error": rel_err,
            "bulk_velocity_error": bulk_err,
            "bulk_numeric": bulk_numeric,
            "bulk_expected": bulk_expected,
        },
    )


def run_couette(nx: int = 40, ny: int = 80, U_top: float = 1.0) -> ValidationResult:
    mesh = structured_mesh(nx, ny)
    pressure = ScalarField("p", mesh, np.zeros(mesh.ncells))
    bcs = [
        NoSlipWall("bottom", mesh, mesh.patch_faces("bottom")),
        MovingWall("top", mesh, mesh.patch_faces("top"), [U_top, 0.0, 0.0]),
    ]
    velocity = solve_momentum(mesh, pressure, bcs)
    ux = velocity.values[:, 0]
    profile = average_by_height(ux, nx, ny)
    H = mesh.spacing[1] * ny
    y = (np.arange(ny) + 0.5) * (H / ny)
    analytic = U_top * y / H
    rel_err = float(np.linalg.norm(profile - analytic) / np.linalg.norm(analytic))
    shear_numeric = (profile[-1] - profile[0]) / H * 0.01
    shear_expected = 0.01 * (U_top / H)
    shear_err = float(abs(shear_numeric - shear_expected) / abs(shear_expected))
    plot_profile("couette_profile.png", y, profile, analytic, "Plane Couette")
    return ValidationResult(
        name="couette",
        metrics={
            "rel_profile_error": rel_err,
            "shear_error": shear_err,
            "shear_numeric": shear_numeric,
            "shear_expected": shear_expected,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation suite")
    parser.add_argument("--lid-Re", type=int, default=100)
    parser.add_argument("--lid-mesh", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "validation_results.json")
    args = parser.parse_args()

    results = []
    results.append(run_lid(args.lid_Re, args.lid_mesh))
    results.append(run_poiseuille())
    results.append(run_couette())

    summary = {item.name: item.metrics for item in results}
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
