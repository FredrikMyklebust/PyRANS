"""k-epsilon turbulence model skeleton."""

from __future__ import annotations

import numpy as np

from .base import TurbulenceModel, register_turbulence


@register_turbulence("kepsilon")
@register_turbulence("k-epsilon")
class KEpsilonModel(TurbulenceModel):
    requires = {"k": True, "epsilon": True, "omega": False}

    def __init__(self, mesh, fields, transport, config=None) -> None:
        super().__init__(mesh, fields, transport, config)
        cfg = self.config
        self.Cmu = float(cfg.get("Cmu", 0.09))
        self.nut_max = float(cfg.get("nutMax", 10.0))
        if "k" in fields:
            self.k = fields["k"]
        else:
            self.k = self._allocate("k", cfg.get("k0", 0.1))
            fields["k"] = self.k
        if "epsilon" in fields:
            self.epsilon = fields["epsilon"]
        else:
            self.epsilon = self._allocate("epsilon", cfg.get("epsilon0", 0.1))
            fields["epsilon"] = self.epsilon

    def _allocate(self, name: str, value: float):
        data = np.full(self.mesh.ncells, float(value))
        from ...core.field import ScalarField  # local import to avoid cycle

        return ScalarField(name, self.mesh, data)

    def correct(self, velocity_field) -> None:
        eps = np.maximum(self.epsilon.values, 1e-8)
        nut = self.Cmu * (self.k.values ** 2) / eps
        np.clip(nut, 0.0, self.nut_max, out=nut)
        self._nut.values[:] = nut
        self._production.values[:] = 0.0
