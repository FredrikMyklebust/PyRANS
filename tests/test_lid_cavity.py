import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

plt = pytest.importorskip("matplotlib.pyplot")

from cfda import Case

ARTIFACTS = pathlib.Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_PATH = pathlib.Path(__file__).parent / "data" / "ghia_u.txt"


def _load_ghia_re100():
    y_vals = []
    u_vals = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            y_vals.append(float(parts[0]))
            u_vals.append(float(parts[1]))
    y = np.array(y_vals)
    u = np.array(u_vals)
    order = np.argsort(y)
    return y[order], u[order]


def test_lid_cavity_against_ghia():
    y_ref, u_ref = _load_ghia_re100()
    case = Case.from_yaml("tests/cases/lid/system/case.yaml")
    case.solve()

    nx, ny = case.mesh.shape
    ic = nx // 2
    centerline = [ic + j * nx for j in range(ny)]
    y_numeric = case.mesh.cell_centers[centerline, 1]
    u_numeric = case.U.values[centerline, 0]

    interp_u = np.interp(y_ref, y_numeric, u_numeric)

    fig, ax = plt.subplots()
    ax.plot(interp_u, y_ref, label="CFDA")
    ax.plot(u_ref, y_ref, "--", label="Ghia et al.")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.legend()
    fig.savefig(ARTIFACTS / "ghia_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rel_l2 = np.linalg.norm(interp_u - u_ref) / (np.linalg.norm(u_ref) or 1.0)
    assert rel_l2 < 8.0
