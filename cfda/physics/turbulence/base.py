"""Base turbulence model implementation and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from ...core.field import ScalarField
from ...core.mesh import Mesh
from ...utils.registry import Registry


turbulence_registry = Registry("turbulence")


def register_turbulence(name: str):
    return turbulence_registry.register(name)


def make_turbulence_model(name: str, *args, **kwargs):
    return turbulence_registry.create(name, *args, **kwargs)


class TurbulenceModel(ABC):
    requires: Dict[str, bool] = {"k": False, "epsilon": False, "omega": False}

    def __init__(
        self,
        mesh: Mesh,
        fields: Dict[str, ScalarField],
        transport,
        config: Optional[Dict],
    ) -> None:
        self.mesh = mesh
        self.fields = fields
        self.transport = transport
        self.config = config or {}
        self._nut = ScalarField("nut", mesh, np.zeros(mesh.ncells))
        self._production = ScalarField("production", mesh, np.zeros(mesh.ncells))

    @abstractmethod
    def correct(self, velocity_field) -> None:
        """Update model internal state for the current iteration."""

    def nut(self) -> ScalarField:
        return self._nut

    def production(self) -> ScalarField:
        return self._production
