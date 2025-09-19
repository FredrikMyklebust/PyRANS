"""Lid-driven cavity benchmark utilities.

Run steady SIMPLE solves for Re = 100/400/1000 on configurable meshes, compare
centerline velocity profiles against the Ghia et al. data set, and produce
plots + error summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda import Case

BASE_CASE = ROOT / "tests" / "cases" / "lid"
DATA_DIR = ROOT / "tests" / "data"
ARTIFACT_DIR = ROOT / "tests" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RE_COLUMNS = [100, 400, 1000, 3200, 5000, 7500, 10000]


def load_ghia_table(filename: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    path = DATA_DIR / filename
    table: Dict[int, Tuple[list[float], list[float]]] = {
        Re: ([], []) for Re in RE_COLUMNS
    }
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(value) for value in line.split()]
            coord = parts[0]
            for col, Re in enumerate(RE_COLUMNS, start=1):
                value = parts[col]
                coord_list, value_list = table[Re]
                coord_list.append(coord)
                value_list.append(value)
    return {
        Re: (np.asarray(coords), np.asarray(values)) for Re, (coords, values) in table.items()
    }


GHIA_U = load_ghia_table("ghia_u.txt")

# Table II (v-velocity along horizontal centreline) transcribed from Ghia et al. 1982
GHIA_V_RAW = {
    100: [
        (0.0000, 0.00000),
        (0.0625, -0.06434),
        (0.1250, -0.12146),
        (0.1875, -0.15662),
        (0.2500, -0.18109),
        (0.3125, -0.19713),
        (0.3750, -0.20673),
        (0.4375, -0.20920),
        (0.5000, -0.20581),
        (0.5625, -0.19677),
        (0.6250, -0.18244),
        (0.6875, -0.16369),
        (0.7500, -0.13230),
        (0.8125, -0.08905),
        (0.8750, -0.03111),
        (0.9375, 0.04602),
        (1.0000, 0.00000),
    ],
    400: [
        (0.0000, 0.00000),
        (0.0625, -0.12115),
        (0.1250, -0.21388),
        (0.1875, -0.27569),
        (0.2500, -0.32627),
        (0.3125, -0.37119),
        (0.3750, -0.40917),
        (0.4375, -0.43643),
        (0.5000, -0.45453),
        (0.5625, -0.46474),
        (0.6250, -0.46801),
        (0.6875, -0.45992),
        (0.7500, -0.42665),
        (0.8125, -0.31966),
        (0.8750, -0.19610),
        (0.9375, -0.06434),
        (1.0000, 0.00000),
    ],
    1000: [
        (0.0000, 0.00000),
        (0.0625, -0.17527),
        (0.1250, -0.32726),
        (0.1875, -0.39017),
        (0.2500, -0.43827),
        (0.3125, -0.47221),
        (0.3750, -0.50000),
        (0.4375, -0.52357),
        (0.5000, -0.54053),
        (0.5625, -0.54302),
        (0.6250, -0.53524),
        (0.6875, -0.50788),
        (0.7500, -0.43829),
        (0.8125, -0.34228),
        (0.8750, -0.23576),
        (0.9375, -0.10313),
        (1.0000, 0.00000),
    ],
}

GHIA_V = {Re: (np.array([p[0] for p in data]), np.array([p[1] for p in data])) for Re, data in GHIA_V_RAW.items()}


@dataclass
class BenchmarkResult:
    Re: int
    mesh: Tuple[int, int]
    iterations: int
    final_U_res: float
    final_mass_res: float
    bulk_speed: float
    u_L2: float
    u_Linf: float
    v_L2: float
    v_Linf: float
    vortex_center: Tuple[float, float]
    sample_points: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "Re": self.Re,
            "mesh": self.mesh,
            "iterations": self.iterations,
            "final_U_res": self.final_U_res,
            "final_mass_res": self.final_mass_res,
            "bulk_speed": self.bulk_speed,
            "u_L2": self.u_L2,
            "u_Linf": self.u_Linf,
            "v_L2": self.v_L2,
            "v_Linf": self.v_Linf,
            "vortex_center": self.vortex_center,
            "sample_points": self.sample_points,
        }


def prepare_case(tmp_root: Path, nx: int, ny: int, reynolds: int, alpha_u: float, alpha_p: float,
                 min_outer: int, max_outer: int, tol_u: float, tol_p: float, tol_mass: float) -> Path:
    shutil.copytree(BASE_CASE, tmp_root, dirs_exist_ok=True)
    case_file = tmp_root / "system" / "case.yaml"
    config = yaml.safe_load(case_file.read_text())
    config["mesh"]["nx"] = nx
    config["mesh"]["ny"] = ny
    config["SIMPLE"]["alphaU"] = alpha_u
    config["SIMPLE"]["alphaP"] = alpha_p
    config["SIMPLE"]["maxOuter"] = max_outer
    config["SIMPLE"]["minOuter"] = min_outer
    config["SIMPLE"]["tolU"] = tol_u
    config["SIMPLE"]["tolP"] = tol_p
    config["SIMPLE"]["tolMass"] = tol_mass
    config["convergence"]["maxIters"] = max_outer
    case_file.write_text(yaml.safe_dump(config))

    transport_file = tmp_root / "constant" / "transport.yaml"
    transport = yaml.safe_load(transport_file.read_text())
    transport["rho"] = 1.0
    transport["mu"] = 1.0 / float(reynolds)
    transport_file.write_text(yaml.safe_dump(transport))
    return case_file


def reshape_field(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((ny, nx))


def compute_centerlines(case: Case) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = case.mesh.shape
    x = reshape_field(case.mesh.cell_centers[:, 0], nx, ny)
    y = reshape_field(case.mesh.cell_centers[:, 1], nx, ny)
    u = reshape_field(case.U.values[:, 0], nx, ny)
    v = reshape_field(case.U.values[:, 1], nx, ny)

    ic = int(np.argmin(np.abs(x[0, :] - 0.5)))
    jc = int(np.argmin(np.abs(y[:, 0] - 0.5)))

    u_line = u[:, ic]
    y_line = y[:, ic]

    v_line = v[jc, :]
    x_line = x[jc, :]

    return y_line, u_line, x_line, v_line


def interpolate_profile(x_numeric: np.ndarray, y_numeric: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    order = np.argsort(x_numeric)
    return np.interp(x_ref, x_numeric[order], y_numeric[order])


def locate_primary_vortex(case: Case) -> Tuple[float, float]:
    nx, ny = case.mesh.shape
    speeds = np.linalg.norm(case.U.values[:, :2], axis=1)
    mask = np.array([len(case.mesh.cell_neighbors[i]) == 4 for i in range(case.mesh.ncells)])
    speeds_masked = np.where(mask, speeds, np.inf)
    index = int(np.argmin(speeds_masked))
    x_c, y_c, _ = case.mesh.cell_centers[index]
    return x_c, y_c


def bulk_kinetic_energy(case: Case) -> float:
    speeds = np.linalg.norm(case.U.values[:, :2], axis=1)
    return 0.5 * np.sum(speeds**2 * case.mesh.cell_volumes)


def run_simulation(Re: int, nx: int, alpha_u: float, alpha_p: float, min_outer: int, max_outer: int,
                   tol_u: float, tol_p: float, tol_mass: float) -> BenchmarkResult:
    with tempfile.TemporaryDirectory(prefix=f"lid_Re{Re}_") as tmp:
        tmp_root = Path(tmp)
        case_path = prepare_case(
            tmp_root,
            nx=nx,
            ny=nx,
            reynolds=Re,
            alpha_u=alpha_u,
            alpha_p=alpha_p,
            min_outer=min_outer,
            max_outer=max_outer,
            tol_u=tol_u,
            tol_p=tol_p,
            tol_mass=tol_mass,
        )
        case = Case.from_yaml(case_path)
        case.solve()

        iterations = len(case.logger.history)
        final_entry = case.logger.history[-1]
        u_res = final_entry.get("U_rel", final_entry.get("U", math.nan))
        mass_res = final_entry.get("mass_norm", final_entry.get("mass", math.nan))

        y_line, u_line, x_line, v_line = compute_centerlines(case)

        # U-profile errors
        y_ref, u_ref = GHIA_U[Re]
        u_interp = interpolate_profile(y_line, u_line, y_ref)
        u_l2 = np.linalg.norm(u_interp - u_ref) / (np.linalg.norm(u_ref) or 1.0)
        u_linf = float(np.max(np.abs(u_interp - u_ref)))

        # V-profile errors
        x_ref, v_ref = GHIA_V[Re]
        v_interp = interpolate_profile(x_line, v_line, x_ref)
        v_l2 = np.linalg.norm(v_interp - v_ref) / (np.linalg.norm(v_ref) or 1.0)
        v_linf = float(np.max(np.abs(v_interp - v_ref)))

        vortex_center = locate_primary_vortex(case)
        bulk_speed = 2.0 * bulk_kinetic_energy(case)

        samples = {
            "u(0.5,0.7344)": float(np.interp(0.7344, y_line, u_line)),
            "u(0.5,0.5000)": float(np.interp(0.5000, y_line, u_line)),
            "u(0.5,0.1016)": float(np.interp(0.1016, y_line, u_line)),
        }

        # Plots
        fig, ax = plt.subplots()
        ax.plot(u_line, y_line, label="CFDA")
        ax.plot(u_ref, y_ref, "o", label="Ghia et al.")
        ax.set_xlabel("u (x=0.5)")
        ax.set_ylabel("y")
        ax.set_title(f"Re={Re}, {nx}x{nx}")
        ax.legend()
        fig.savefig(ARTIFACT_DIR / f"lid_Re{Re}_u_{nx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot(x_line, v_line, label="CFDA")
        ax.plot(x_ref, v_ref, "o", label="Ghia et al.")
        ax.set_xlabel("x")
        ax.set_ylabel("v (y=0.5)")
        ax.set_title(f"Re={Re}, {nx}x{nx}")
        ax.legend()
        fig.savefig(ARTIFACT_DIR / f"lid_Re{Re}_v_{nx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return BenchmarkResult(
            Re=Re,
            mesh=(nx, nx),
            iterations=iterations,
            final_U_res=u_res,
            final_mass_res=mass_res,
            bulk_speed=bulk_speed,
            u_L2=float(u_l2),
            u_Linf=u_linf,
            v_L2=float(v_l2),
            v_Linf=v_linf,
            vortex_center=vortex_center,
            sample_points=samples,
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lid-driven cavity benchmarks")
    parser.add_argument("--Re", type=int, nargs="*", default=[100, 400, 1000])
    parser.add_argument("--mesh", type=int, default=64, help="Number of cells per side")
    parser.add_argument("--alphaU", type=float, default=0.3)
    parser.add_argument("--alphaP", type=float, default=0.3)
    parser.add_argument("--min-outer", type=int, default=50)
    parser.add_argument("--max-outer", type=int, default=400)
    parser.add_argument("--tolU", type=float, default=1e-7)
    parser.add_argument("--tolP", type=float, default=1e-7)
    parser.add_argument("--tolMass", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "lid_benchmark.json")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    results = []
    for Re in args.Re:
        result = run_simulation(
            Re=Re,
            nx=args.mesh,
            alpha_u=args.alphaU,
            alpha_p=args.alphaP,
            min_outer=args.min_outer,
            max_outer=args.max_outer,
            tol_u=args.tolU,
            tol_p=args.tolP,
            tol_mass=args.tolMass,
        )
        results.append(result)
        print(json.dumps(result.to_dict(), indent=2))

    args.output.write_text(
        json.dumps([result.to_dict() for result in results], indent=2)
    )


if __name__ == "__main__":
    main()
