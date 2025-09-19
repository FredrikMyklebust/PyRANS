import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda import Case


CASES = [
    (
        "tests/cases/simple/system/case.yaml",
        {"residual_max": 1.0, "u_norm_min": 1e-3},
    ),
    (
        "tests/cases/lid/system/case.yaml",
        {"residual_max": 30.0, "u_norm_min": 1e-2, "ux_max_min": 0.5},
    ),
    (
        "tests/cases/channel/system/case.yaml",
        {"residual_max": 1.0e7, "u_norm_min": 1e-2, "ux_mean_abs_min": 0.05},
    ),
]


@pytest.mark.parametrize("case_path, expectations", CASES)
def test_case_convergence(case_path, expectations):
    case = Case.from_yaml(case_path)
    case.solve()

    assert np.isfinite(case.U.values).all()
    assert np.isfinite(case.p.values).all()

    assert case.logger.history
    residuals = [
        entry.get("p_rel")
        if entry.get("p_rel") is not None
        else entry.get("mass_norm", 0.0)
        for entry in case.logger.history
    ]
    assert residuals[-1] < expectations["residual_max"]

    u_norm = float(np.linalg.norm(case.U.values))
    assert u_norm > expectations.get("u_norm_min", 0.0)

    if "ux_max_min" in expectations:
        assert case.U.values[:, 0].max() > expectations["ux_max_min"]

    if "ux_mean_abs_min" in expectations:
        assert abs(case.U.values[:, 0].mean()) > expectations["ux_mean_abs_min"]
