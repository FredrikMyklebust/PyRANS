import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt = pytest.importorskip("matplotlib.pyplot")

from cfda.core.field import ScalarField, VectorField
from cfda.core.mesh import Mesh
from cfda.core.bc import NoSlipWall, MovingWall
from cfda.physics.transport import ConstantTransport
from cfda.physics.turbulence.laminar import LaminarModel
from cfda.solvers.momentum import MomentumAssembler

ARTIFACTS = pathlib.Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _structured_mesh(nx: int, ny: int) -> Mesh:
    return Mesh.structured(
        nx,
        ny,
        lengths=(1.0, 1.0),
        patch_aliases={"ymin": "top", "ymax": "bottom", "xmin": "inlet", "xmax": "outlet"},
    )


def _solve_momentum(mesh: Mesh, pressure: ScalarField, velocity_bcs) -> VectorField:
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


def _average_by_height(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    profiles = np.zeros(ny)
    for j in range(ny):
        profiles[j] = values[j * nx : (j + 1) * nx].mean()
    return profiles


def _plot_profile(filename: str, y: np.ndarray, numeric: np.ndarray, analytic: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(numeric, y, label="CFDA")
    ax.plot(analytic, y, "--", label="Analytic")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    path = ARTIFACTS / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _poiseuille_setup(G: float = 1.0, nx: int = 40, ny: int = 80) -> dict:
    mesh = _structured_mesh(nx, ny)
    centers = mesh.cell_centers[:, 0]
    pressure = ScalarField("p", mesh, -G * centers)
    bcs = [
        NoSlipWall("bottom", mesh, mesh.patch_faces("bottom")),
        NoSlipWall("top", mesh, mesh.patch_faces("top")),
    ]
    velocity = _solve_momentum(mesh, pressure, bcs)
    ux = velocity.values[:, 0]
    profile = _average_by_height(ux, nx, ny)
    H = mesh.spacing[1] * ny
    y = (np.arange(ny) + 0.5) * (H / ny)
    mu = 0.01
    analytic = (G / (2 * mu)) * y * (H - y)
    left_cells = np.array([j * nx for j in range(ny)])
    right_cells = np.array([j * nx + (nx - 1) for j in range(ny)])
    p_left = pressure.values[left_cells].mean()
    p_right = pressure.values[right_cells].mean()
    Lx = mesh.spacing[0] * nx
    dpdx = (p_right - p_left) / Lx
    bulk_numeric = ux.mean()
    bulk_expected = (-dpdx) * H**2 / (12 * mu)
    return {
        "mesh": mesh,
        "y": y,
        "numeric": profile,
        "analytic": analytic,
        "bulk_numeric": bulk_numeric,
        "bulk_expected": bulk_expected,
        "dpdx": dpdx,
    }


def _couette_setup(U_top: float = 1.0, nx: int = 40, ny: int = 80) -> dict:
    mesh = _structured_mesh(nx, ny)
    pressure = ScalarField("p", mesh, np.zeros(mesh.ncells))
    bcs = [
        NoSlipWall("bottom", mesh, mesh.patch_faces("bottom")),
        MovingWall("top", mesh, mesh.patch_faces("top"), [U_top, 0.0, 0.0]),
    ]
    velocity = _solve_momentum(mesh, pressure, bcs)
    ux = velocity.values[:, 0]
    profile = _average_by_height(ux, nx, ny)
    H = mesh.spacing[1] * ny
    y = (np.arange(ny) + 0.5) * (H / ny)
    analytic = U_top * y / H
    mu = 0.01
    shear_numeric = (profile[-1] - profile[0]) / (H - 0.0) * mu
    shear_expected = mu * (U_top / H)
    return {
        "mesh": mesh,
        "y": y,
        "numeric": profile,
        "analytic": analytic,
        "shear_numeric": shear_numeric,
        "shear_expected": shear_expected,
    }


def test_poiseuille_profile():
    data = _poiseuille_setup()
    rel_err = np.linalg.norm(data["numeric"] - data["analytic"]) / np.linalg.norm(data["analytic"])
    assert rel_err < 0.1

    bulk_err = abs(data["bulk_numeric"] - data["bulk_expected"]) / abs(data["bulk_expected"])
    assert bulk_err < 0.1

    _plot_profile("poiseuille_profile.png", data["y"], data["numeric"], data["analytic"], "Plane Poiseuille")


def test_couette_profile():
    data = _couette_setup()
    rel_err = np.linalg.norm(data["numeric"] - data["analytic"]) / np.linalg.norm(data["analytic"])
    assert rel_err < 0.05

    shear_err = abs(data["shear_numeric"] - data["shear_expected"]) / abs(data["shear_expected"])
    assert shear_err < 0.05

    _plot_profile("couette_profile.png", data["y"], data["numeric"], data["analytic"], "Couette Flow")
