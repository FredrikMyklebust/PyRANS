"""Profiling helper for benchmark cases.

Usage:
    python scripts/profile_cases.py \
        --case tests/cases/lid/system/case.yaml \
        --case tests/cases/channel/system/case.yaml

The script enables coarse-grained profiling inside the SIMPLE/PISO/PIMPLE
coupling agents (currently detailed timings are implemented for SIMPLE).  After
each case runs it prints a timing breakdown and dumps a JSON summary next to the
console output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfda import Case


def run_case(case_path: Path) -> dict[str, float]:
    case = Case.from_yaml(case_path)
    coupling = case.coupling
    if hasattr(coupling, "enable_profiling"):
        coupling.enable_profiling(True)
    start = perf_counter()
    case.solve()
    total_wall = perf_counter() - start
    timings = coupling.get_timings() if hasattr(coupling, "get_timings") else {}
    if "total" not in timings:
        timings["total"] = total_wall
    timings["wall_clock"] = total_wall
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile selected CFD cases")
    parser.add_argument(
        "--case",
        action="append",
        type=Path,
        help="Path to case.yaml (repeat for multiple cases)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/artifacts/profiling_results.json"),
        help="Where to write the aggregated JSON timings",
    )
    args = parser.parse_args()

    case_paths = args.case or [
        Path("tests/cases/lid/system/case.yaml"),
        Path("tests/cases/channel/system/case.yaml"),
    ]

    summary: dict[str, dict[str, float]] = {}
    for path in case_paths:
        print(f"\n=== Profiling {path} ===")
        timings = run_case(path)
        total = timings.get("total", timings.get("wall_clock", 0.0)) or 1.0
        for key, value in sorted(timings.items(), key=lambda kv: kv[1], reverse=True):
            if key == "outer_iterations":
                print(f"{key:>20}: {value:.0f}")
            else:
                pct = 100.0 * value / total
                print(f"{key:>20}: {value:8.4f} s ({pct:5.1f}%)")
        summary[str(path)] = timings

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote profiling summary to {args.output}")


if __name__ == "__main__":
    main()
