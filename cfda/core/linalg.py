"""Linear algebra scaffolding for FV matrices."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:  # Optional dependency for multigrid pressure solves
    import pyamg  # type: ignore
except ImportError:  # pragma: no cover - optional path
    pyamg = None

try:
    from scipy import sparse
except ImportError as exc:  # pragma: no cover - SciPy is required for AMG path
    sparse = None  # type: ignore

from .mesh import Mesh


class FvMatrix:
    """Sparse matrix builder for finite-volume systems."""

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self._diag = np.zeros(mesh.ncells)
        self._offdiag: Dict[Tuple[int, int], float] = {}
        self._rhs = np.zeros(mesh.ncells)

    def add_diag(self, cell_ids: Iterable[int], coeffs: Iterable[float]) -> None:
        for cid, a_p in zip(cell_ids, coeffs):
            self._diag[cid] += a_p

    def add_nb(
        self, cell_ids: Iterable[int], nb_ids: Iterable[int], coeffs: Iterable[float]
    ) -> None:
        for cid, nid, coeff in zip(cell_ids, nb_ids, coeffs):
            if cid == nid:
                self._diag[cid] += coeff
            else:
                key = (cid, nid)
                self._offdiag[key] = self._offdiag.get(key, 0.0) + coeff

    def add_rhs(self, cell_ids: Iterable[int], values: Iterable[float]) -> None:
        for cid, val in zip(cell_ids, values):
            self._rhs[cid] += val

    def to_dense(self) -> np.ndarray:
        n = self.mesh.ncells
        mat = np.zeros((n, n))
        np.fill_diagonal(mat, self._diag)
        for (row, col), coeff in self._offdiag.items():
            mat[row, col] += coeff
        return mat

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector)
        result = self._diag * vec
        for (row, col), coeff in self._offdiag.items():
            result[row] += coeff * vec[col]
        return result

    def diagonal(self) -> np.ndarray:
        return self._diag.copy()

    def rhs(self) -> np.ndarray:
        return self._rhs.copy()

    def apply_dirichlet(self, cell_id: int, value: float, rhs: np.ndarray) -> None:
        # Set the row to identity
        for (row, col), coeff in list(self._offdiag.items()):
            if row == cell_id:
                del self._offdiag[(row, col)]
        self._diag[cell_id] = 1.0
        rhs[cell_id] = value

    def solve(
        self,
        rhs: np.ndarray | None = None,
        method: str = "direct",
        tol: float = 1e-10,
        maxiter: int = 500,
        return_stats: bool = False,
        initial_guess: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
        method = method.lower()
        amg_methods = {"amg", "gamg"}
        iterative_methods = {"cg", "bicgstab"}
        valid_methods = {"direct", *iterative_methods, *amg_methods}
        if method not in valid_methods:
            raise NotImplementedError(f"Unknown solver method '{method}'")

        b = self._rhs if rhs is None else rhs

        if method == "direct":
            A = self.to_dense()
            if initial_guess is None:
                x0 = np.zeros_like(b)
            else:
                x0 = np.asarray(initial_guess, dtype=float)
            r0 = A @ x0 - b
            initial_res = float(np.linalg.norm(r0))
            solution = np.linalg.solve(A, b)
            if not return_stats:
                return solution
            residual_vec = self.matvec(solution) - b
            final_res = float(np.linalg.norm(residual_vec))
            denom = initial_res if initial_res > 0.0 else float(np.linalg.norm(b)) or 1.0
            stats = {
                "initial": initial_res,
                "final": final_res,
                "relative": final_res / denom,
                "iterations": 1.0,
            }
            return solution, stats

        if method in amg_methods:
            if pyamg is None or sparse is None:  # pragma: no cover - import guard
                raise RuntimeError(
                    "AMG solver requested but pyamg/scipy are not available."
                )

            A = self.to_csr()
            ml = (
                pyamg.ruge_stuben_solver(A)
                if method == "gamg"
                else pyamg.smoothed_aggregation_solver(A)
            )
            x0 = None if initial_guess is None else np.asarray(initial_guess, dtype=float)
            residuals: List[float] = []
            solution = ml.solve(b, x0=x0, tol=tol, maxiter=maxiter, residuals=residuals)

            if residuals:
                initial_res = float(residuals[0])
                final_res = float(residuals[-1])
            else:
                initial_res = float(np.linalg.norm(b if x0 is None else b - self.matvec(x0)))
                residual_vec = self.matvec(solution) - b
                final_res = float(np.linalg.norm(residual_vec))

            denom = initial_res if initial_res > 0.0 else float(np.linalg.norm(b)) or 1.0
            stats = {
                "initial": initial_res,
                "final": final_res,
                "relative": final_res / denom,
                "iterations": float(len(residuals)) if residuals else float("nan"),
            }
            if return_stats:
                return np.asarray(solution), stats
            return np.asarray(solution)

        diag = self.diagonal()
        inv_diag = np.zeros_like(diag)
        mask = diag != 0.0
        inv_diag[mask] = 1.0 / diag[mask]

        if initial_guess is None:
            x0 = np.zeros_like(b)
        else:
            x0 = np.asarray(initial_guess, dtype=float)

        x = x0.copy()

        residual = b - self.matvec(x)
        initial_res = float(np.linalg.norm(residual))
        if initial_res == 0.0:
            if return_stats:
                return x, {"initial": 0.0, "final": 0.0, "relative": 0.0, "iterations": 0.0}
            return x

        tol_abs = tol
        tol_rel = tol * initial_res
        tol_target = max(tol_abs, tol_rel)
        maxiter = max(1, maxiter)

        if method == "cg":
            z = inv_diag * residual
            p = z.copy()
            rz_old = np.dot(residual, z)
            for iteration in range(1, maxiter + 1):
                Ap = self.matvec(p)
                denom = np.dot(p, Ap)
                if denom == 0.0:
                    break
                alpha = rz_old / denom
                x += alpha * p
                residual -= alpha * Ap
                res_norm = float(np.linalg.norm(residual))
                if res_norm <= tol_target:
                    break
                z = inv_diag * residual
                rz_new = np.dot(residual, z)
                beta = rz_new / rz_old
                p = z + beta * p
                rz_old = rz_new
            else:
                iteration = float(maxiter)
        else:  # BiCGSTAB
            r_tilde = residual.copy()
            rho_old = alpha = omega = 1.0
            v = np.zeros_like(residual)
            p_vec = np.zeros_like(residual)
            for iteration in range(1, maxiter + 1):
                rho_new = np.dot(r_tilde, residual)
                if rho_new == 0.0:
                    break
                if iteration == 1:
                    p_vec = residual.copy()
                else:
                    beta = (rho_new / rho_old) * (alpha / omega)
                    p_vec = residual + beta * (p_vec - omega * v)

                y = inv_diag * p_vec
                v = self.matvec(y)
                denom = np.dot(r_tilde, v)
                if denom == 0.0:
                    break
                alpha = rho_new / denom
                s = residual - alpha * v
                s_norm = float(np.linalg.norm(s))
                if s_norm <= tol_target:
                    x += alpha * y
                    residual = s
                    break

                z = inv_diag * s
                t = self.matvec(z)
                tt = np.dot(t, t)
                if tt == 0.0:
                    break
                omega = np.dot(t, s) / tt
                x += alpha * y + omega * z
                residual = s - omega * t
                res_norm = float(np.linalg.norm(residual))
                if res_norm <= tol_target:
                    break
                if omega == 0.0:
                    break
                rho_old = rho_new
            else:
                iteration = float(maxiter)

        final_res = float(np.linalg.norm(residual))
        denom = initial_res if initial_res > 0.0 else float(np.linalg.norm(b)) or 1.0
        stats_data = {
            "initial": initial_res,
            "final": final_res,
            "relative": final_res / denom,
            "iterations": float(iteration),
        }
        if method == "cg" and (
            not np.isfinite(final_res)
            or final_res > max(tol_abs, tol_rel) * 10.0
        ):
            return self.solve(
                rhs=b,
                method="bicgstab",
                tol=tol,
                maxiter=maxiter,
                return_stats=return_stats,
                initial_guess=initial_guess,
            )

        if return_stats:
            return x, stats_data
        return x

    def to_csr(self):  # pragma: no cover - exercised via AMG path
        if sparse is None:
            raise RuntimeError("SciPy is required for CSR conversion")
        n = self.mesh.ncells
        rows = []
        cols = []
        data = []
        for i, val in enumerate(self._diag):
            if val != 0.0:
                rows.append(i)
                cols.append(i)
                data.append(val)
        for (row, col), coeff in self._offdiag.items():
            rows.append(row)
            cols.append(col)
            data.append(coeff)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    def clear_rhs(self) -> None:
        self._rhs[:] = 0.0

    def copy_structure(self) -> "FvMatrix":
        dup = FvMatrix(self.mesh)
        dup._diag = self._diag.copy()
        dup._offdiag = dict(self._offdiag)
        return dup
