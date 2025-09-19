"""Case management for the CFD library."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..core.bc import (
    MovingWall,
    NoSlipWall,
    PressureOutlet,
    VelocityInlet,
    ZeroGradient,
)
from ..core.field import ScalarField, VectorField
from ..core.mesh import Mesh
from ..physics.turbulence import make_turbulence_model
from ..physics.transport import ConstantTransport
from ..solvers.coupling.base import make_coupling
from ..solvers.momentum import MomentumAssembler
from ..solvers.pressure import PressureAssembler
from ..utils.io import read_yaml_file
from ..utils.logging import IterationLogger
from .time import TimeControl


VELOCITY_BCS = {
    "noslip": NoSlipWall,
    "movingwall": MovingWall,
    "velocityinlet": VelocityInlet,
}

PRESSURE_BCS = {
    "pressureoutlet": PressureOutlet,
    "fixedpressure": PressureOutlet,
    "zerogradient": ZeroGradient,
}


class Case:
    def __init__(self, root: Path, config: Dict) -> None:
        self.root = root
        self.config = config
        self.mesh = self._build_mesh(config.get("mesh", {}))
        self.relaxation = config.get("relaxation", {"U": 0.7, "p": 0.3})
        self.convergence = config.get(
            "convergence", {"U": 1e-6, "p": 1e-6, "maxIters": 50}
        )
        self.transport = self._load_transport()
        self.fields: Dict[str, ScalarField | VectorField] = {}
        self.boundary_data: Dict[str, Dict] = {}
        self._load_primary_fields()
        self.time_control = TimeControl.from_dict(config.get("time"))
        self.logger = IterationLogger(config.get("Coupling", "SIMPLE"))
        self.residual_control = {
            key.lower(): float(value)
            for key, value in (config.get("residualControl", {}) or {}).items()
        }

        self.velocity_bcs = self._build_velocity_bcs()
        self.pressure_bcs = self._build_pressure_bcs()

        self.momentum_assembler = MomentumAssembler(self.mesh, self.velocity_bcs)
        self.pressure_reference = self._resolve_pressure_reference(config.get("pressureReference"))
        self.pressure_assembler = PressureAssembler(
            self.mesh, self.pressure_bcs, self.pressure_reference
        )

        turbulence_cfg = read_yaml_file(self.root / "constant" / "turbulence.yaml")
        model_name = turbulence_cfg.get("TurbulenceModel", "Laminar")
        model_config = turbulence_cfg.get(model_name, {})
        self.turbulence_model = make_turbulence_model(
            model_name,
            mesh=self.mesh,
            fields={k: v for k, v in self.fields.items() if isinstance(v, ScalarField)},
            transport=self.transport,
            config=model_config,
        )

        coupling_name = config.get("Coupling", "SIMPLE")
        coupling_cfg = config.get(coupling_name, {})
        from ..solvers.coupling import simple, simplec, piso, pimple  # noqa: F401

        self.coupling = make_coupling(coupling_name, self, coupling_cfg)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Case":
        case_path = Path(path)
        if case_path.name.lower() != "case.yaml":
            raise ValueError("Expected system/case.yaml")
        root = case_path.parent.parent
        config = read_yaml_file(case_path)
        return cls(root=root, config=config)

    def _build_mesh(self, mesh_cfg: Dict) -> Mesh:
        mtype = mesh_cfg.get("type", "structured").lower()
        if mtype != "structured":
            raise NotImplementedError("Only structured meshes are supported in the MVP")
        nx = int(mesh_cfg.get("nx", 10))
        ny = int(mesh_cfg.get("ny", 10))
        lengths = tuple(mesh_cfg.get("lengths", [1.0, 1.0]))
        patches = mesh_cfg.get("patches")
        return Mesh.structured(nx, ny, lengths=lengths, patch_aliases=patches)

    def _load_transport(self) -> ConstantTransport:
        path = self.root / "constant" / "transport.yaml"
        data = read_yaml_file(path)
        return ConstantTransport.from_dict(data)

    def _parse_uniform(self, entry, vector: bool):
        if isinstance(entry, dict) and "uniform" in entry:
            entry = entry["uniform"]
        if isinstance(entry, str):
            tokens = entry.replace("uniform", "").replace("[", " ").replace("]", " ").split()
            entry = [float(tok.strip(",")) for tok in tokens]
        values = np.asarray(entry, dtype=float)
        if vector:
            if values.size == 1:
                values = np.repeat(values, 3)
            values = values.reshape(1, -1)
            return np.repeat(values, self.mesh.ncells, axis=0)
        return np.full(self.mesh.ncells, float(values.reshape(-1)[0]))

    def _load_field_file(self, name: str, vector: bool):
        path = self.root / "0" / f"{name}.yaml"
        if not path.exists():
            return None
        data = read_yaml_file(path)
        internal = data.get("internalField", 0.0 if not vector else [0.0, 0.0, 0.0])
        values = self._parse_uniform(internal, vector)
        field = VectorField(name, self.mesh, values) if vector else ScalarField(name, self.mesh, values)
        self.fields[name] = field
        self.boundary_data[name] = data.get("boundaryField", {})
        return field

    def _load_primary_fields(self) -> None:
        self.U = self._load_field_file("U", vector=True)
        if self.U is None:
            raise RuntimeError("Velocity field U is required")
        self.p = self._load_field_file("p", vector=False)
        if self.p is None:
            raise RuntimeError("Pressure field p is required")
        # Optional turbulence scalars
        for name in ("k", "epsilon", "omega"):
            self._load_field_file(name, vector=False)

    def _build_velocity_bcs(self):
        bcs = []
        bc_data = self.boundary_data.get("U", {})
        for patch, cfg in bc_data.items():
            faces = self.mesh.patch_faces(patch)
            info = cfg or {}
            if isinstance(info, str):
                info = {"type": info}
            bc_type = info.get("type", "noSlip")
            cls = VELOCITY_BCS.get(bc_type.lower())
            if cls is None:
                continue
            if cls is VelocityInlet or cls is MovingWall:
                value = info.get("value", [0.0, 0.0, 0.0])
                bc = cls(patch, self.mesh, faces, value)
            else:
                bc = cls(patch, self.mesh, faces)
            bcs.append(bc)
        return bcs

    def _build_pressure_bcs(self):
        bcs = []
        bc_data = self.boundary_data.get("p", {})
        for patch, cfg in bc_data.items():
            faces = self.mesh.patch_faces(patch)
            info = cfg or {}
            if isinstance(info, str):
                info = {"type": info}
            bc_type = info.get("type", "pressureOutlet")
            cls = PRESSURE_BCS.get(bc_type.lower())
            if cls is None:
                continue
            value = info.get("value", 0.0)
            if cls is PressureOutlet:
                bc = cls(patch, self.mesh, faces, value)
            else:
                bc = cls(patch, self.mesh, faces)
            bcs.append(bc)
        return bcs

    def _resolve_pressure_reference(self, cfg):
        nx, ny = self.mesh.shape
        if not cfg:
            return (0, 0.0)
        if "cell" in cfg:
            cell = cfg["cell"]
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                raise ValueError("pressureReference.cell must be [i, j]")
            i, j = int(cell[0]), int(cell[1])
            if not (0 <= i < nx and 0 <= j < ny):
                raise ValueError("pressureReference.cell out of bounds")
            cell_id = j * nx + i
        elif "cellIndex" in cfg:
            cell_id = int(cfg["cellIndex"])
            if not (0 <= cell_id < self.mesh.ncells):
                raise ValueError("pressureReference.cellIndex out of bounds")
        else:
            cell_id = 0
        value = float(cfg.get("value", 0.0))
        return (cell_id, value)

    def solve(self):
        for time in self.time_control:
            self.transport.update(time)
            self.turbulence_model.correct(self.U)
            self.coupling.solve_step(self)

    def field(self, name: str):
        return self.fields[name]
