#!/usr/bin/env python3
"""Diagnostic probe for momentum assembly sanity.

Builds two simple Stokes configurations and verifies geometry metrics,
BC-imposed face velocities, and momentum coefficients are finite and
positive before any solve. This helps track down NaNs/zeros that later
poison SIMPLE/PISO/PIMPLE loops.
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda.core import fv_ops
from cfda.core.bc.inlet import VelocityInlet
from cfda.core.bc.wall import MovingWall, NoSlipWall
from cfda.core.field import ScalarField, VectorField
from cfda.core.mesh import Mesh
from cfda.solvers.momentum import MomentumAssembler

EPS = 1e-30


def _finite(arr) -> bool:
    """True if every entry in arr is finite."""
    return np.isfinite(arr).all()


def check_geometry(mesh: Mesh) -> None:
    """Ensure basic geometric metrics are well-defined."""
    for fid, face in enumerate(mesh.faces):
        area_mag = float(np.linalg.norm(face.area_vector))
        assert area_mag > 0.0 and np.isfinite(area_mag), f"|Sf| invalid at face {fid}"
        if face.neighbour is not None:
            d_vec = mesh.cell_centers[face.neighbour] - mesh.cell_centers[face.owner]
            d_mag = float(np.linalg.norm(d_vec))
            assert d_mag > EPS and np.isfinite(d_mag), f"dPN invalid at face {fid}"


def check_face_velocity(mesh: Mesh, vel_bcs, U: VectorField) -> None:
    """Interpolate and stamp boundary velocities, then ensure finiteness."""
    face_vel = fv_ops.interpolate(mesh, U.values)
    for bc in vel_bcs:
        bc.update_face_values(face_vel, U)
    assert _finite(face_vel), "NaNs in face velocity after BC stamping"


def build_and_check(title: str, vel_bcs, alpha_u: float = 0.7, mu: float = 1.0) -> None:
    """Build a momentum system and assert coefficient sanity."""
    print(f"\n=== {title} ===")
    mesh = Mesh.structured(16, 16)
    U = VectorField("U", mesh, np.zeros((mesh.ncells, 3)))
    p = ScalarField("p", mesh, np.zeros(mesh.ncells))

    check_geometry(mesh)
    check_face_velocity(mesh, vel_bcs, U)

    assembler = MomentumAssembler(mesh, vel_bcs)
    system = assembler.build(
        velocity=U,
        pressure=p,
        rho=1.0,
        mu=mu,
        nut=None,
        alpha_u=max(alpha_u, 1e-6),
    )

    a_p = system.diag
    r_au = system.rAU
    r_auf = system.rAUf

    assert _finite(a_p) and np.all(a_p > 0.0), (
        f"aP invalid: min={float(np.nanmin(a_p)):.3e}, max={float(np.nanmax(a_p)):.3e}"
    )
    assert _finite(r_au) and np.all(r_au > 0.0), "rAU has non-positive/NaN entries"
    assert _finite(r_auf) and np.all(r_auf > 0.0), "rAUf has non-positive/NaN entries"

    print(f"aP min/max   = {float(a_p.min()):.3e} / {float(a_p.max()):.3e}")
    print(f"rAU min/max  = {float(r_au.min()):.3e} / {float(r_au.max()):.3e}")
    print(f"rAUf min/max = {float(r_auf.min()):.3e} / {float(r_auf.max()):.3e}")
    print("MOMENTUM NAN PROBE PASS ✅")


def main() -> None:
    mesh = Mesh.structured(16, 16)
    top = mesh.patch_faces("ymax")
    bottom = mesh.patch_faces("ymin")
    left = mesh.patch_faces("xmin")
    right = mesh.patch_faces("xmax")

    # Case A: all walls no-slip
    vel_bcs_a = [
        NoSlipWall("ymax", mesh, top),
        NoSlipWall("ymin", mesh, bottom),
        NoSlipWall("xmin", mesh, left),
        NoSlipWall("xmax", mesh, right),
    ]
    build_and_check("Case A — all no-slip (diffusion only)", vel_bcs_a)

    # Case B: moving lid
    vel_bcs_b = [
        MovingWall("ymax", mesh, top, [1.0, 0.0, 0.0]),
        NoSlipWall("ymin", mesh, bottom),
        NoSlipWall("xmin", mesh, left),
        NoSlipWall("xmax", mesh, right),
    ]
    build_and_check("Case B — moving lid (Stokes cavity)", vel_bcs_b)

    # Case C: inlet/outlet with moving lid (matches simple.yaml)
    vel_bcs_c = [
        VelocityInlet("xmin", mesh, left, [1.0, 0.0, 0.0]),
        VelocityInlet("xmax", mesh, right, [1.0, 0.0, 0.0]),
        MovingWall("ymax", mesh, top, [1.0, 0.0, 0.0]),
        NoSlipWall("ymin", mesh, bottom),
    ]
    build_and_check("Case C — inlet/outlet + moving lid", vel_bcs_c)


if __name__ == "__main__":
    main()
