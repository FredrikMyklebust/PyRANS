# CFDA – Modular PIMPLE-style Python CFD framework

This repository hosts a finite-volume CFD playground inspired by OpenFOAM’s
runtime-polymorphic architecture.  It implements collocated SIMPLE/PISO/PIMPLE
coupling, Rhie–Chow flux correction, interchangeable turbulence models, and a
collection of diagnostics/benchmarks for plane flows and the lid-driven cavity.

The project structure mirrors `agents.md`, the original design brief.

```
cfda/
  core/          # mesh, fields, FV operators, linear algebra
  numerics/      # interpolation schemes and Rhie–Chow support
  physics/       # transport properties and turbulence plug-ins
  solvers/       # momentum/pressure assemblers and coupling agents
  run/           # Case loading and time control
scripts/         # diagnostics, benchmarks, validation runners
tests/           # unit/regression tests and reference data
```

## Environment

The code targets Python 3.11+.  A minimal environment is described in
`environment.yml` (NumPy, SciPy, matplotlib, pyamg for optional multigrid).

```
conda env create -f environment.yml
conda activate cfda
```

## Running the main diagnostics

### Lid-driven cavity benchmark

```
python scripts/lid_benchmark.py --Re 100 --mesh 32 \
    --min-outer 60 --max-outer 120 \
    --alphaU 0.25 --alphaP 0.3 \
    --tolU 1e-7 --tolP 1e-7 --tolMass 1e-5
```

This produces centreline comparison plots (`tests/artifacts/lid_Re*_*.png`)
and a JSON summary in the same directory.

### Validation suite (lid + Poiseuille + Couette)

```
python scripts/validation_suite.py
```

The script emits:
- `tests/artifacts/validation_results.json` with residual/error metrics.
- Profile plots for Poiseuille and Couette flows for quick visual checks.

### Pre-flight FV diagnostics

```
python scripts/pre_lid_gate.py
```

This gate exercises geometry sanity checks, manufactured Laplace/Poisson
solutions, and a Stokes cavity to confirm the Rhie–Chow pipeline before running
SIMPLE/PISO/PIMPLE loops.

## Running the regression tests

```
pytest -q
```

The suite covers:
- Component tests for SIMPLE building blocks (`tests/test_simple_components.py`).
- Plane Poiseuille/Couette analytic comparisons with plots.
- Regression cases for the lid-driven cavity (`tests/test_lid_cavity.py`).
- Case-level end-to-end solves for simple/lid/channel dictionaries
  (`tests/test_simple_case.py`).

## Channel setups

`tests/cases/channel/system/case.yaml` demonstrates a SIMPLE channel flow with
inlet velocity, zero-gradient outlet velocity, and mixed pressure BCs.  Run it
standalone via:

```
python - <<'PY'
from pathlib import Path
from cfda import Case
case = Case.from_yaml(Path('tests/cases/channel/system/case.yaml'))
case.solve()
PY
```

## Producing additional plots

- `tests/artifacts/` is intentionally ignored (`.gitignore`) so the validation
  scripts can regenerate figures without polluting git history.
- The benchmark/validation scripts automatically drop PNGs/JSON summaries in
  this directory; copy them out if you need to keep snapshots.

## Contributing

1. Run `pytest -q` before sending changes.
2. Use the validation suite to sanity-check lid/plane-flow behaviour.
3. Document new scripts in this README.

Feel free to extend the scheme registry, turbulence plug-ins, or add new
benchmark cases—everything is runtime-discoverable via the registries defined in
`cfda/physics` and `cfda/solvers/coupling`.
