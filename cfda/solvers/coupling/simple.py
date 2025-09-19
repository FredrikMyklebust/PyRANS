"""SIMPLE coupling agent."""

from __future__ import annotations

import numpy as np

from ...core import fv_ops
from ...core.fv_ops import grad
from ..momentum import MomentumAssembler
from ..pressure import PressureAssembler
from .base import CouplingAgent, register_coupling
from time import perf_counter


@register_coupling("simple")
class SimpleCoupling(CouplingAgent):
    def __init__(self, case, config=None) -> None:
        super().__init__(case, config)
        cfg = self.config
        self.alpha_u = float(cfg.get("alphaU", case.relaxation.get("U", 0.7)))
        self.alpha_p = float(cfg.get("alphaP", case.relaxation.get("p", 0.3)))
        self.alpha_p_min = float(cfg.get("alphaPMin", 0.05))
        self.adaptive_alpha_p = bool(cfg.get("adaptiveAlphaP", True))
        self._alpha_p_dynamic = self.alpha_p
        self.max_outer = int(cfg.get("maxOuter", case.convergence.get("maxIters", 50)))
        self.min_outer = int(cfg.get("minOuter", 2))
        self.n_pressure = int(cfg.get("nPressureCorrectors", 1))
        self.n_nonorth = int(cfg.get("nNonOrthogonalCorrectors", 0))
        self.u_tol = float(cfg.get("tolU", case.convergence.get("U", 1e-6)))
        self.p_tol = float(cfg.get("tolP", case.convergence.get("p", 1e-6)))
        self.mass_tol = float(cfg.get("tolMass", case.convergence.get("p", 1e-6)))
        self.solver_u = cfg.get("solverU", "bicgstab")
        self.solver_u_tol = float(cfg.get("solverTolU", 1e-10))
        self.solver_u_maxiter = int(cfg.get("solverMaxIterU", 500))
        self.solver_p = cfg.get("solverP", "cg")
        self.solver_p_tol = float(cfg.get("solverTolP", 1e-10))
        self.solver_p_maxiter = int(cfg.get("solverMaxIterP", 500))
        self.residual_control = case.residual_control
        self.cumulative_mass = 0.0
        self.initial_abs = {}
        self.beta_phi_base = float(cfg.get("betaPhi", 0.7))
        self._prev_mass_norm: float | None = None

    def solve_step(self, case) -> None:
        mesh = case.mesh
        volumes = case.mesh.cell_volumes

        if self.profiling:
            self.reset_timings()
            solve_start = perf_counter()

        for outer in range(1, self.max_outer + 1):
            outer_start = perf_counter() if self.profiling else None
            tic = perf_counter() if self.profiling else None
            momentum = case.momentum_assembler.build(
                velocity=case.U,
                pressure=case.p,
                rho=case.transport.density(),
                mu=case.transport.viscosity(),
                nut=case.turbulence_model.nut(),
                alpha_u=self.alpha_u,
            )
            if self.profiling and tic is not None:
                self.timings["momentum_assembly"] += perf_counter() - tic

            u_rel_res = 0.0
            u_initial = 0.0
            u_final = 0.0
            for comp in range(3):
                prev = case.U.values[:, comp].copy()
                tic = perf_counter() if self.profiling else None
                solution, stats = momentum.matrices[comp].solve(
                    momentum.rhs[:, comp],
                    return_stats=True,
                    method=self.solver_u,
                    tol=self.solver_u_tol,
                    maxiter=self.solver_u_maxiter,
                    initial_guess=prev,
                )
                case.U.values[:, comp] = solution
                u_rel_res = max(u_rel_res, stats["relative"])
                u_initial = max(u_initial, stats["initial"])
                u_final = max(u_final, stats["final"])
                if self.profiling and tic is not None:
                    self.timings["momentum_solve"] += perf_counter() - tic

            phi_star = momentum.phi.copy()
            phi = phi_star.copy()
            mass_before = float(
                np.sqrt(np.sum(volumes * fv_ops.div(mesh, phi) ** 2))
            )
            p_initial = 0.0
            p_final = 0.0
            p_rel_res = 0.0

            rho = case.transport.density()
            n_corr = max(1, self.n_pressure + self.n_nonorth)
            alpha_sequence = [1.0] * n_corr
            if self.n_pressure > 0:
                alpha_sequence[-1] = (
                    self._alpha_p_dynamic if self.adaptive_alpha_p else self.alpha_p
                )
            corr_masses: list[float] = []
            has_dirichlet = any(
                bc.is_dirichlet() for bc in case.pressure_assembler.pressure_bcs
            )
            beta_phi = self.beta_phi_base
            if not has_dirichlet:
                beta_phi = max(beta_phi, 1.0)

            print(
                "rAU range:",
                float(momentum.rAU.min()),
                float(momentum.rAU.max()),
            )
            print(
                "rAUf range:",
                float(momentum.rAUf.min()),
                float(momentum.rAUf.max()),
            )
            print(
                "momentum diag range:",
                float(momentum.diag.min()),
                float(momentum.diag.max()),
            )

            assert np.all(momentum.rAU > 0.0), "rAU has non-positive entries"
            assert np.all(momentum.rAUf > 0.0), "rAUf has non-positive entries"

            centers = mesh.cell_centers
            converged_outer = False

            for corr_idx, alpha_iter in enumerate(alpha_sequence, start=1):
                pressure_system = case.pressure_assembler.build(
                    phi,
                    pressure=case.p,
                    rho=rho,
                    alpha_p=alpha_iter,
                    rAUf=momentum.rAUf,
                )

                if not np.shares_memory(pressure_system.rAUf, momentum.rAUf):
                    raise AssertionError(
                        "Pressure assembler did not receive shared rAUf array"
                    )

                print(
                    f"corrector {corr_idx}: rAUf checksum matrix path =",
                    float(np.sum(momentum.rAUf)),
                )

                phi_prev = phi.copy()
                p_corr, p_stats = pressure_system.matrix.solve(
                    pressure_system.rhs,
                    return_stats=True,
                    method=self.solver_p,
                    tol=self.solver_p_tol,
                    maxiter=self.solver_p_maxiter,
                )

                case.p.values += alpha_iter * p_corr
                p_initial = max(p_initial, p_stats["initial"])
                p_final = max(p_final, p_stats["final"])
                p_rel_res = max(p_rel_res, p_stats["relative"])

                grad_pcorr = grad(
                    mesh,
                    p_corr,
                    case.pressure_assembler.pressure_bcs,
                )
                for comp in range(3):
                    case.U.values[:, comp] -= momentum.rAU * grad_pcorr[:, comp]

                print(
                    f"corrector {corr_idx}: rAUf checksum flux path =",
                    float(np.sum(momentum.rAUf)),
                )

                flux_tic = perf_counter() if self.profiling else None
                delta_phi = np.zeros_like(phi)
                for fid, face in enumerate(mesh.faces):
                    neigh = face.neighbour
                    if neigh is None:
                        bc = case.pressure_assembler._face_bc.get(fid)
                        if bc is None or not bc.is_dirichlet():
                            continue
                        owner = face.owner
                        vec = face.center - centers[owner]
                        distance = float(np.linalg.norm(vec))
                        if distance <= 0.0:
                            continue
                        n_hat = vec / distance
                        s_proj = float(np.dot(face.area_vector, n_hat))
                        face_value = float(getattr(bc, "pressure", p_corr[owner]))
                        dpdn = (face_value - float(p_corr[owner])) * (s_proj / distance)
                        correction = beta_phi * momentum.rAUf[fid] * dpdn
                        phi[fid] -= correction
                        delta_phi[fid] = correction
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
                    correction = beta_phi * momentum.rAUf[fid] * dpdn
                    phi[fid] -= correction
                    delta_phi[fid] = correction

                phi_expected = phi_prev - delta_phi
                if not np.allclose(
                    phi, phi_expected, rtol=1e-9, atol=1e-12
                ):
                    diff = np.abs(phi - phi_expected)
                    bad = [
                        (fid, float(diff_val), float(phi[fid]), float(phi_expected[fid]))
                        for fid, diff_val in enumerate(diff)
                        if diff_val > 1e-9
                    ]
                    print("phi mismatch details (first 5):", bad[:5])
                    raise AssertionError(
                        f"phi update mismatch (orientation/reset/double-count) max diff {diff.max():.3e}"
                    )

                err = 0.0
                mag = 0.0
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
                    rhs = dp * (s_proj / distance)
                    lhs = (
                        delta_phi[fid]
                        / max(beta_phi * momentum.rAUf[fid], 1e-30)
                    )
                    err += (lhs - rhs) ** 2
                    mag += lhs**2 + rhs**2
                rel_misfit = np.sqrt(err / max(mag, 1e-30))
                print(f"corrector {corr_idx}: grad·Sf vs Δp/|d| misfit = {rel_misfit:.3e}")
                assert rel_misfit < 1e-10, "gradient operator / geometry mismatch"

                assert np.isfinite(phi).all()
                assert np.isfinite(case.U.values).all()
                assert np.isfinite(case.p.values).all()

                div_phi_corr = fv_ops.div(mesh, phi)
                corr_mass = float(np.sqrt(np.sum(volumes * div_phi_corr**2)))
                corr_masses.append(corr_mass)
                if corr_idx == 1:
                    print(
                        f"mass drop (corr #1): {mass_before:.3e} -> {corr_mass:.3e}"
                    )

                no_flux_velocity = {
                    bc.name
                    for bc in case.momentum_assembler.velocity_bcs
                    if bc.__class__.__name__.lower() in {"noslipwall", "movingwall"}
                }
                zero_flux_patches = {
                    bc.name
                    for bc in case.pressure_assembler.pressure_bcs
                    if not bc.is_dirichlet() and bc.name in no_flux_velocity
                }
                for name in mesh.patches():
                    s = float(sum(phi[f] for f in mesh.patch_faces(name)))
                    print(f"patch {name} sum(phi) = {s:.3e}")
                    if name in zero_flux_patches:
                        assert abs(s) < 1e-12, f"boundary flux injected on patch {name}"

                if self.profiling and flux_tic is not None:
                    self.timings["flux_correction"] += perf_counter() - flux_tic

                div_phi = fv_ops.div(mesh, phi)
                mass_after = float(np.sqrt(np.sum(volumes * div_phi**2)))
                global_mass = float(np.sum(phi))
                self.cumulative_mass += global_mass
                mass_corr_last = corr_masses[-1]
                mass_corr_max = max(corr_masses)

                if self.adaptive_alpha_p:
                    if self._prev_mass_norm is not None and mass_after > self._prev_mass_norm * 1.5:
                        self._alpha_p_dynamic = max(
                            self._alpha_p_dynamic * 0.5, self.alpha_p_min
                        )
                    elif mass_after < (self._prev_mass_norm or mass_after):
                        self._alpha_p_dynamic = min(
                            self._alpha_p_dynamic * 1.05, self.alpha_p
                        )

                self._prev_mass_norm = mass_after

                turb_tic = perf_counter() if self.profiling else None
                case.turbulence_model.correct(case.U)
                if self.profiling and turb_tic is not None:
                    self.timings["turbulence_correct"] += perf_counter() - turb_tic

                if outer == 1:
                    self.initial_abs = {
                        "u": u_initial,
                        "p": p_initial,
                        "phi": mass_after,
                    }

                case.logger.log(
                    outer,
                    {
                        "U_initial": u_initial,
                        "U_final": u_final,
                        "U_rel": u_rel_res,
                        "p_initial": p_initial,
                        "p_final": p_final,
                        "p_rel": p_rel_res,
                        "mass_before": mass_before,
                        "mass_norm": mass_after,
                        "mass_global": global_mass,
                        "mass_cumulative": self.cumulative_mass,
                        "mass_corr_last": mass_corr_last,
                        "mass_corr_max": mass_corr_max,
                    },
                )

                converged_outer = self._check_convergence(
                    u_rel_res, u_final, p_rel_res, p_final, mass_after
                )
                if outer >= self.min_outer and converged_outer:
                    break

            if self.profiling and outer_start is not None:
                self.timings["outer_loop"] += perf_counter() - outer_start

        if self.profiling:
            self.timings["outer_iterations"] = outer
            self.timings["total"] = perf_counter() - solve_start

    def _check_convergence(self, u_rel, u_final, p_rel, p_final, mass_norm):
        rc = self.residual_control
        u_ok = u_rel < self.u_tol
        if "u" in rc:
            u_ok = u_final < rc["u"]
        p_ok = p_rel < self.p_tol
        if "p" in rc:
            p_ok = p_final < rc["p"]
        m_ok = mass_norm < self.mass_tol
        if "phi" in rc:
            m_ok = mass_norm < rc["phi"]
        return u_ok and p_ok and m_ok
