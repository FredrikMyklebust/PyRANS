"""PIMPLE coupling agent."""

from __future__ import annotations

import numpy as np
from time import perf_counter

from ...core import fv_ops
from ...core.fv_ops import grad
from .base import CouplingAgent, register_coupling


@register_coupling("pimple")
class PimpleCoupling(CouplingAgent):
    def __init__(self, case, config=None) -> None:
        super().__init__(case, config)
        cfg = self.config
        self.n_outer = int(cfg.get("nOuterCorrectors", 2))
        self.n_pressure = int(cfg.get("nPressureCorrectors", 2))
        self.alpha_u = float(cfg.get("alphaU", case.relaxation.get("U", 0.7)))
        self.alpha_p = float(cfg.get("alphaP", case.relaxation.get("p", 0.3)))
        self.solver_u = cfg.get("solverU", "bicgstab")
        self.solver_u_tol = float(cfg.get("solverTolU", 1e-10))
        self.solver_u_maxiter = int(cfg.get("solverMaxIterU", 500))
        self.solver_p = cfg.get("solverP", "cg")
        self.solver_p_tol = float(cfg.get("solverTolP", 1e-10))
        self.solver_p_maxiter = int(cfg.get("solverMaxIterP", 500))
        self.cumulative_mass = 0.0
        self.u_tol = float(cfg.get("tolU", case.convergence.get("U", 1e-6)))
        self.p_tol = float(cfg.get("tolP", case.convergence.get("p", 1e-6)))
        self.residual_control = case.residual_control
        self.initial_abs = {}

    def solve_step(self, case) -> None:
        mesh = case.mesh
        volumes = mesh.cell_volumes
        if self.profiling:
            self.reset_timings()
            solve_start = perf_counter()
        for outer in range(self.n_outer):
            U_old = case.U.values.copy()
            momentum = case.momentum_assembler.build(
                velocity=case.U,
                pressure=case.p,
                rho=case.transport.density(),
                mu=case.transport.viscosity(),
                nut=case.turbulence_model.nut(),
                alpha_u=self.alpha_u,
            )
            u_rel_res = 0.0
            u_initial = 0.0
            u_final = 0.0
            for comp in range(3):
                prev = case.U.values[:, comp].copy()
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

            phi = momentum.phi.copy()
            mass_before = float(np.sqrt(np.sum(volumes * fv_ops.div(mesh, phi) ** 2)))
            rho = case.transport.density()
            p_rel_res = 0.0
            p_initial = 0.0
            p_final = 0.0
            corr_masses: list[float] = []
            beta_phi = 0.7
            for corr in range(self.n_pressure):
                alpha_p = self.alpha_p if corr == self.n_pressure - 1 else 1.0
                pressure_system = case.pressure_assembler.build(
                    phi,
                    pressure=case.p,
                    rho=rho,
                    alpha_p=alpha_p,
                    rAUf=momentum.rAUf,
                )
                if not np.shares_memory(pressure_system.rAUf, momentum.rAUf):
                    raise AssertionError(
                        "Pressure assembler did not preserve rAUf sharing (PIMPLE)"
                    )
                mass_imbalance = float(np.linalg.norm(pressure_system.rhs))
                p_corr, p_stats = pressure_system.matrix.solve(
                    pressure_system.rhs,
                    return_stats=True,
                    method=self.solver_p,
                    tol=self.solver_p_tol,
                    maxiter=self.solver_p_maxiter,
                )
                case.p.values += alpha_p * p_corr
                p_initial = max(p_initial, p_stats["initial"])
                p_final = max(p_final, p_stats["final"])
                p_rel_res = max(p_rel_res, p_stats["relative"])
                grad_pcorr = grad(
                    mesh,
                    p_corr,
                    case.pressure_assembler.pressure_bcs,
                )
                face_pcorr = fv_ops.interpolate(mesh, p_corr)
                for bc in case.pressure_assembler.pressure_bcs:
                    bc.update_face_values(face_pcorr, p_corr)
                for comp in range(3):
                    case.U.values[:, comp] -= momentum.rAU * grad_pcorr[:, comp]
                self._apply_flux_correction(
                    mesh,
                    phi,
                    momentum.rAUf,
                    p_corr,
                    face_pcorr,
                    case.pressure_assembler._face_bc,
                    beta_phi,
                )

                assert np.isfinite(phi).all()
                assert np.isfinite(case.U.values).all()
                assert np.isfinite(case.p.values).all()

                div_phi_corr = fv_ops.div(mesh, phi)
                corr_masses.append(float(np.sqrt(np.sum(volumes * div_phi_corr**2))))

            div_phi = fv_ops.div(mesh, phi)
            mass_after = float(np.sqrt(np.sum(volumes * div_phi**2)))
            global_mass = float(np.sum(phi))
            self.cumulative_mass += global_mass
            mass_corr_last = corr_masses[-1] if corr_masses else mass_after
            mass_corr_max = max(corr_masses) if corr_masses else mass_after

            case.turbulence_model.correct(case.U)
            delta_u = case.U.values - U_old
            u_change = float(np.linalg.norm(delta_u) / (np.linalg.norm(case.U.values) or 1.0))
            if outer == 0:
                self.initial_abs = {
                    "u": u_initial,
                    "p": p_initial,
                    "phi": mass_after,
                }
            case.logger.log(
                outer + 1,
                {
                    "U_initial": u_initial,
                    "U_final": u_final,
                    "U_rel": max(u_rel_res, u_change),
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
            if self._check_convergence(u_rel_res, u_final, p_rel_res, p_final, mass_after):
                break
        if self.profiling:
            self.timings["total"] = perf_counter() - solve_start

    def _apply_flux_correction(
        self,
        mesh,
        phi,
        rAUf,
        p_corr,
        face_pcorr,
        face_bc,
        beta_phi,
    ) -> None:
        centers = mesh.cell_centers
        for fid, face in enumerate(mesh.faces):
            neigh = face.neighbour
            owner = face.owner
            if neigh is None:
                bc = face_bc.get(fid) if face_bc is not None else None
                if bc is None or not bc.is_dirichlet():
                    continue
                vec = face.center - centers[owner]
                distance = float(np.linalg.norm(vec))
                if distance <= 0.0:
                    continue
                n_hat = vec / distance
                s_proj = float(np.dot(face.area_vector, n_hat))
                delta_p = float(face_pcorr[fid] - p_corr[owner])
                dpdn = delta_p * (s_proj / distance)
                phi[fid] -= beta_phi * rAUf[fid] * dpdn
                continue

            vec = centers[neigh] - centers[owner]
            distance = float(np.linalg.norm(vec))
            if distance <= 0.0:
                continue
            n_hat = vec / distance
            s_proj = float(np.dot(face.area_vector, n_hat))
            delta_p = float(p_corr[neigh] - p_corr[owner])
            dpdn = delta_p * (s_proj / distance)
            phi[fid] -= beta_phi * rAUf[fid] * dpdn

    def _check_convergence(self, u_rel, u_final, p_rel, p_final, mass_norm):
        rc = self.residual_control
        u_ok = u_rel < self.u_tol
        if "u" in rc:
            u_ok = u_final < rc["u"]
        p_ok = p_rel < self.p_tol
        if "p" in rc:
            p_ok = p_final < rc["p"]
        m_ok = True
        if "phi" in rc:
            m_ok = mass_norm < rc["phi"]
        return u_ok and p_ok and m_ok
