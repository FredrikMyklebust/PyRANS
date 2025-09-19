# agents.md — Modular PIMPLE‑style Python CFD framework

A design spec for a Python CFD library that mirrors OpenFOAM’s modularity while following the SIMPLE/PISO/PIMPLE family of algorithms. The goal is to let you swap solvers, discretisations, linear solvers, and turbulence models with minimal code changes.

---

## 0) Goals & non‑goals

**Goals**
- Finite‑Volume (FV) core with collocated storage and Rhie–Chow fluxes.
- Pressure–velocity coupling via interchangeable strategies: `SIMPLE`, `SIMPLEC`, `PISO`, and `PIMPLE` (transient SIMPLE with inner PISO loops).
- OpenFOAM‑like runtime polymorphism for turbulence models, schemes, and boundary conditions.
- Clear separation between **numerics** (discretisation/solvers) and **physics** (models, properties, source terms).

**Non‑goals**
- General unstructured mesh I/O (initially focus on structured Cartesian for clarity; keep interfaces mesh‑agnostic).
- GPU acceleration in v1 (keep numerics vectorisable; add backends later).

---

## 1) Package layout (proposed)

```
cfda/
  core/
    mesh.py           # Mesh, Cell, Face, Topology
    field.py          # ScalarField, VectorField (collocated)
    fv_ops.py         # div, grad, laplacian, interpolate, faceFlux
    linalg.py         # Matrix, CSR, linear solvers (CG/AMG wrappers)
    relax.py          # Under‑relaxation control
    bc/
      base.py         # BoundaryCondition
      wall.py         # noSlip, movingWall
      inlet.py        # fixedValue (velocity‑inlet)
      outlet.py       # fixedPressure (pressure‑outlet)
  numerics/
    schemes.py        # Upwind, Linear, TVD variants
    rhie_chow.py      # Collocated flux correction
  physics/
    transport.py      # rho(T), mu(T), etc. (constant initially)
    turbulence/
      base.py         # TurbulenceModel (RANS base)
      laminar.py      # Laminar (nut = 0)
      kepsilon.py     # k‑ε
      komegassg.py    # k‑ω SST
  solvers/
    momentum.py       # Assembler for U‑equations
    pressure.py       # Pressure correction assembler
    coupling/
      base.py         # CouplingAgent (interface)
      simple.py       # SIMPLE
      simplec.py      # SIMPLEC
      piso.py         # PISO (1+ pressure corrections)
      pimple.py       # PIMPLE (outer SIMPLE, inner PISO)
  run/
    case.py           # Case, dictionaries, run‑time selection (registry)
    time.py           # Time/step control
  utils/
    io.py             # I/O, dict/YAML parsing
    logging.py        # Iteration monitors, residuals
```

---

## 2) Core abstractions

### 2.1 Fields & mesh
- `ScalarField(name, mesh, values)`
- `VectorField(name, mesh, values)` → shape `(Ncells, 3)`; components accessible as `.x, .y, .z`.
- `Mesh(cells, faces)` → faces have `owner`, `neighbour`, `areaVector (Af * nHat)`, `center`.

### 2.2 Finite‑volume operators
- `grad(phi)` → cell‑centred gradient via Gauss theorem.
- `div(phi_u)` → divergence of a face flux.
- `laplacian(gamma, phi)` → diffusion operator.
- `interpolate(phi, scheme)` → cell→face; default Linear; upwind for advection.
- `faceFlux(U)` → mass flux `mf = rho * (Af · U_f)`; **always** Rhie–Chow‑corrected.

### 2.3 Linear system API
```python
class FvMatrix:
    def add_diag(self, cell_ids, aP): ...
    def add_nb(self, cell_ids, nb_ids, aNb): ...
    def add_rhs(self, cell_ids, b): ...
    def solve(self, method="AMG", tol=1e-8, maxiter=200): ...
```

### 2.4 Boundary conditions
```python
class BoundaryCondition(ABC):
    def apply_coeffs(self, A: FvMatrix, b: np.ndarray, field: ScalarField|VectorField): ...
    def update_face_values(self, field): ...  # computes face values consistent with BC
```
Built‑ins: `NoSlipWall`, `MovingWall(U)`, `VelocityInlet(U)`, `PressureOutlet(p)`.

---

## 3) Turbulence model plugin system

OpenFOAM‑style runtime selection via a registry:
```python
class TurbulenceModel(ABC):
    requires = {"k": False, "omega": False, "epsilon": False}
    def __init__(self, mesh, fields, props, dict): ...
    def correct(self, U: VectorField): ...            # update model state per outer loop/time step
    def nut(self) -> ScalarField: ...                 # turbulent viscosity ν_t
    def production(self) -> ScalarField: ...          # P_k if applicable
```
Examples:
- `LaminarModel` → `nut() = 0`.
- `KEpsilonModel`, `KOmegaSSTModel` with sub‑fields (`k`, `epsilon`/`omega`) as `ScalarField`s and their own transport equations assembled using the same FV API.

Swap models by case dict:
```yaml
TurbulenceModel: KEpsilon
kEpsilonCoeffs:
  Cmu: 0.09
  C1: 1.44
  C2: 1.92
```

---

## 4) Momentum & pressure assemblers

### 4.1 Momentum (for U)
`MomentumAssembler.build(U, p, rho, mu, mf)` returns `FvMatrix A(U) * U = H - grad(p)` with implicit relaxation support.

Key steps:
1) For each face, compute advection nb‑coeff using upwind on `mf` and diffusion nb‑coeff using `mu` and geometric spacing.
2) Build diagonal `aP` and add relaxation `aP_relax = aP / alphaU`. Add relaxation source `S_relaxU` so the relaxed system preserves the unrelaxed fixed point.
3) Add boundary forces per BC type (wall, velocity‑inlet, pressure‑outlet).

### 4.2 Pressure correction assembler
`PressureAssembler.build(mf, aU_relax, H_over_aUrelax)` builds `aPP p' + Σ aPN p'_N = − m_source` and returns matrix + rhs.
- `m_source` from face‑mass imbalance computed with pseudo‑velocity (H/aU_relax) and Rhie–Chow face interpolation.

---

## 5) Coupling agents (interchangeable)

All implement:
```python
class CouplingAgent(ABC):
    def solve_step(self, case): ...  # one outer loop (steady) or one dt (transient)
```

### 5.1 SIMPLE
**Outer loop**:
1) Assemble & solve `U` with current `p` (implicit relaxation on U).
2) Build pressure‑correction eqn from continuity using pseudo‑velocity.
3) Solve for `p'`, update `p := p + alphaP * p'` (pressure relaxation), and correct `U` and `mf`.
4) `turbulence.correct(U)`.
5) Check residuals; repeat until converged.

### 5.2 SIMPLEC
As SIMPLE, but uses modified velocity‑correction consistent with SIMPLEC to reduce the need for pressure relaxation.

### 5.3 PISO (transient focus)
Within a time step:
1) Predict `U` from momentum with `p^n`.
2) Pressure correction #1 → correct `p, U, mf`.
3) (Optional) Pressure correction #2 (rebuild source with updated fields) → improved `p` without under‑relaxation.

### 5.4 PIMPLE
For large dt / stiff cases: outer SIMPLE‑like loops, each loop contains 1–2 PISO pressure corrections.

Runtime selection example:
```yaml
Coupling: PIMPLE
PIMPLE:
  nOuterCorrectors: 3
  nPressureCorrectors: 2
  alphaU: 0.7
  alphaP: 0.3
```

---

## 6) Rhie–Chow‑corrected collocated fluxes

Provide a dedicated utility `RhieChowFlux(U, p, aUrelax, H)` that returns face velocity `Uf` and mass flux `mf`:
- Linear interpolate cell velocities, then **replace** pressure‑gradient contribution with a pressure‑Poisson‑consistent form based on `H/aUrelax`.
- This removes checkerboarding and ties continuity to the pressure correction equation.

API:
```python
def rhie_chow_face_velocity(face, cells, U, p, H_over_aUrelax): ...
```

---

## 7) Under‑relaxation & convergence control
- `alphaU` implicit (diagonal scaling) with companion source.
- `alphaP` explicit on pressure update; PISO disables pressure relaxation.
- Residual computation per equation; stopping criteria configured in case dict.

```yaml
relaxation:
  U: 0.7
  p: 0.3
convergence:
  U: 1e-6
  p: 1e-6
  maxIters: 200
```

---

## 8) Time integration
- `steady`: iterate coupling agent until residuals fall below tolerance.
- `transient`: `for t in timeGrid:` → assemble transient terms in momentum (and turbulence equations), then call coupling agent per dt. Choose `Cr`‑safe dt or sub‑cycle.

---

## 9) Minimal solver loop pseudo‑code (PIMPLE)
```python
case = Case.from_yaml("system/case.yaml")
U, p, rho, mu = case.fields()
model = TurbulenceModelRegistry.make(case.turbulence, case)
coupling = CouplingRegistry.make(case.coupling, case)

for t in case.time:
    case.transport.update(t)              # rho/mu if variable
    model.correct(U)                      # pre‑predictor model update
    coupling.solve_step(case)             # does inner SIMPLE/PISO loops
    case.write_if_needed(t)
```

`CouplingRegistry.make("PIMPLE")` internally:
```python
def solve_step(case):
    for outer in range(nOuterCorrectors):
        A_U, H, aUrelax = MomentumAssembler.build(...)
        U[:] = A_U.solve(H - grad(p))
        for k in range(nPressureCorrectors):
            mf = faceFlux(U, p, aUrelax, H)   # Rhie–Chow corrected
            A_p, rhs = PressureAssembler.build(mf, aUrelax, H/aUrelax)
            p_corr = A_p.solve(rhs)
            p += alphaP * p_corr               # alphaP=1 in PISO
            U, mf = correctVelocityAndFlux(U, mf, p_corr, aUrelax)
        model.correct(U)
        if converged(): break
```

---

## 10) Turbulence model example: k‑ε skeleton
```python
class KEpsilon(TurbulenceModel):
    requires = {"k": True, "epsilon": True}
    def __init__(self, mesh, fields, props, dict):
        self.k = fields.get('k')
        self.epsilon = fields.get('epsilon')
        self.Cmu, self.C1, self.C2, self.sigm_k, self.sigm_e = ...
    def nut(self):
        return clamp(self.Cmu * (self.k**2) / self.epsilon, 0, nutMax)
    def correct(self, U):
        # assemble & solve k and epsilon transport equations using fv_ops
        # use production = nut * (2*S:S)
        pass
```

---

## 11) Dictionary format (OpenFOAM‑like)
```
/system
  case.yaml
/constant
  transport.yaml      # rho, mu
  turbulence.yaml     # model name & coeffs
/0
  U.yaml              # initial & BCs
  p.yaml
  k.yaml, epsilon.yaml (if needed)
```

Example `U.yaml`:
```yaml
internalField: uniform [0,0,0]
boundaryField:
  inlet:  velocityInlet { value: [1,0,0] }
  outlet: pressureOutlet {}
  walls:  noSlip {}
```

---

## 12) Testing checklist
- Lid‑driven cavity (Re 100, 1k) → steady SIMPLE vs literature.
- Channel flow → pressure gradient vs bulk velocity.
- Transient start‑up flow → PISO stability; αP=1.
- Toggle turbulence models: Laminar → k‑ε; verify ν_t>0 and sensitivity.

---

## 13) Extensibility hooks
- `@register_turbulence("Name")` decorator for models.
- `@register_coupling("SIMPLE"|"PISO"|...)` for coupling agents.
- `@register_scheme("Linear"|"Upwind"|...)` for interpolation.
- Replaceable linear solver backends via entry points.

---

## 14) Deliverables (MVP)
- `cfda` package with modules above.
- Example cases and parity tests.
- Docs/diagrams of data‑flow between assemblers and coupling agents.

---

## 15) Notes & citations (implementation guidance)
- Use collocated storage + Rhie–Chow to avoid checkerboard pressure/velocity.
- Apply under‑relaxation implicitly for momentum and (usually) explicitly for pressure.
- For transient, prefer PISO with 1–2 pressure corrections per `dt`; for steady, prefer SIMPLE/SIMPLEC; for stiff transients use PIMPLE.

