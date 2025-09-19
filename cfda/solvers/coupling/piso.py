"""PISO coupling agent."""

from __future__ import annotations

import numpy as np

from ...core import fv_ops
from ...core.fv_ops import grad
from .base import CouplingAgent, register_coupling


@register_coupling("piso")
class PISOCoupling(CouplingAgent):
    def __init__(self, case, config=None) -> None:
        super().__init__(case, config)
        cfg = self.config
        self.n_pressure = int(cfg.get("nPressureCorrectors", 2))
        self.solver_u = cfg.get("solverU", "bicgstab")
        self.solver_u_tol = float(cfg.get("solverTolU", 1e-10))
        self.solver_u_maxiter = int(cfg.get("solverMaxIterU", 500))
        self.solver_p = cfg.get("solverP", "cg")
        self.solver_p_tol = float(cfg.get("solverTolP", 1e-10))
        self.solver_p_maxiter = int(cfg.get("solverMaxIterP", 500))
        self.cumulative_mass = 0.0
        self.cumulative_mass = 0.0

    def solve_step(self, case) -> None:
        mesh = case.mesh
        volumes = mesh.cell_volumes
        momentum = case.momentum_assembler.build(
            velocity=case.U,
            pressure=case.p,
            rho=case.transport.density(),
            mu=case.transport.viscosity(),
            nut=case.turbulence_model.nut(),
            alpha_u=1.0,
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
        p_initial = 0.0
        p_final = 0.0
        p_rel_res = 0.0
        beta_phi = 0.7
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
        assert np.all(momentum.rAU > 0.0), "rAU has non-positive entries (PISO)"
        assert np.all(momentum.rAUf > 0.0), "rAUf has non-positive entries (PISO)"
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
                raise AssertionError("Pressure assembler did not preserve rAUf sharing (PISO)")
            print(
                f"PISO corrector {corr+1}: rAUf checksum matrix path =",
                float(np.sum(momentum.rAUf)),
            )
            phi_prev = phi.copy()
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
            print(
                f"PISO corrector {corr+1}: rAUf checksum flux path =",
                float(np.sum(momentum.rAUf)),
            )
            self._apply_flux_correction(
                mesh,
                phi,
                momentum.rAUf,
                p_corr,
                face_pcorr,
                case.pressure_assembler._face_bc,
                beta_phi,
            )

            delta_phi = np.zeros_like(phi)
            err = 0.0
            mag = 0.0
            for fid, face in enumerate(mesh.faces):
                neigh = face.neighbour
                owner = face.owner
                if neigh is None:
                    bc = case.pressure_assembler._face_bc.get(fid)
                    if bc is None or not bc.is_dirichlet():
                        continue
                    vec = face.center - mesh.cell_centers[owner]
                    distance = float(np.linalg.norm(vec))
                    if distance <= 0.0:
                        continue
                    n_hat = vec / distance
                    s_proj = float(np.dot(face.area_vector, n_hat))
                    dpdn = float(face_pcorr[fid] - p_corr[owner]) * (
                        s_proj / distance
                    )
                else:
                    vec = mesh.cell_centers[neigh] - mesh.cell_centers[owner]
                    distance = float(np.linalg.norm(vec))
                    if distance <= 0.0:
                        continue
                    n_hat = vec / distance
                    s_proj = float(np.dot(face.area_vector, n_hat))
                    dpdn = float(p_corr[neigh] - p_corr[owner]) * (
                        s_proj / distance
                    )
                delta_phi[fid] = -beta_phi * momentum.rAUf[fid] * dpdn
                lhs = (-delta_phi[fid]) / max(beta_phi * momentum.rAUf[fid], 1e-30)
                rhs = dpdn
                err += (lhs - rhs) ** 2
                mag += lhs ** 2 + rhs ** 2
            phi_expected = phi_prev + delta_phi
            if not np.allclose(phi, phi_expected, rtol=1e-9, atol=1e-12):
                raise AssertionError("PISO: phi update mismatch")
            rel_misfit = np.sqrt(err / max(mag, 1e-30))
            print(f"PISO corrector {corr+1} gradient misfit: {rel_misfit:.3e}")
            assert rel_misfit < 1e-10, "PISO gradient/geometry mismatch"

            assert np.isfinite(phi).all()
            assert np.isfinite(case.U.values).all()
            assert np.isfinite(case.p.values).all()
            zero_flux_patches = {
                bc.name for bc in case.pressure_assembler.pressure_bcs if not bc.is_dirichlet()
            }
            for name in mesh.patches():
                s = float(sum(phi[f] for f in mesh.patch_faces(name)))
                print(f"PISO patch {name} sum(phi) = {s:.3e}")
                if name in zero_flux_patches:
                    assert abs(s) < 1e-12, f"PISO boundary flux injected on patch {name}"

        div_phi = fv_ops.div(mesh, phi)
        mass_after = float(np.sqrt(np.sum(volumes * div_phi**2)))
        global_mass = float(np.sum(phi))
        self.cumulative_mass += global_mass

        case.turbulence_model.correct(case.U)
        case.logger.log(
            1,
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
            },
        )

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
