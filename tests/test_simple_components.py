import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda.core.mesh import Mesh
from cfda.core.field import ScalarField, VectorField
from cfda.numerics.rhie_chow import rhie_chow_face_velocity
from cfda.physics.transport import ConstantTransport
from cfda.physics.turbulence.laminar import LaminarModel
from cfda.solvers.momentum import MomentumAssembler
from cfda.solvers.pressure import PressureAssembler


def test_momentum_assembler_diffusion_matrix():
    mesh = Mesh.structured(2, 2)
    velocity = VectorField("U", mesh, np.zeros((mesh.ncells, 3)))
    pressure = ScalarField("p", mesh, mesh.cell_centers[:, 0])

    transport = ConstantTransport(rho=1.0, mu=0.01)
    model = LaminarModel(mesh, {}, transport, {})

    assembler = MomentumAssembler(mesh, velocity_bcs=[])
    system = assembler.build(
        velocity=velocity,
        pressure=pressure,
        rho=transport.density(),
        mu=transport.viscosity(),
        nut=model.nut(),
        alpha_u=1.0,
    )

    diag = system.matrices[0]._diag
    offdiag = system.matrices[0]._offdiag
    rhs = system.rhs[:, 0]

    assert np.allclose(diag, 0.02)
    assert all(np.isclose(value, -0.01) for value in offdiag.values())
    assert np.allclose(rhs, -0.125)


def test_rhie_chow_zero_gradient_returns_zero_velocity():
    mesh = Mesh.structured(2, 2)
    velocity = VectorField("U", mesh, np.zeros((mesh.ncells, 3)))
    pressure = ScalarField("p", mesh, np.zeros(mesh.ncells))
    a_diag = np.ones(mesh.ncells)
    H_over_a = np.zeros((mesh.ncells, 3))

    face_vel = rhie_chow_face_velocity(mesh, velocity, pressure, a_diag, H_over_a)
    assert np.allclose(face_vel, 0.0)


def test_pressure_assembler_zero_mass_flux_zero_rhs():
    mesh = Mesh.structured(2, 2)
    pressure = ScalarField("p", mesh, np.zeros(mesh.ncells))
    face_flux = np.zeros(len(mesh.faces))

    assembler = PressureAssembler(mesh, pressure_bcs=[])
    system = assembler.build(
        face_flux=face_flux,
        pressure=pressure,
        rho=1.0,
        alpha_p=1.0,
    )

    assert np.allclose(system.mass_flux, 0.0)
    assert np.allclose(system.rhs, 0.0)
