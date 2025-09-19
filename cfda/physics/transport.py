"""Transport properties models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ConstantTransport:
    rho: float = 1.0
    mu: float = 1.0e-3

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, float]]) -> "ConstantTransport":
        data = data or {}
        return cls(rho=float(data.get("rho", 1.0)), mu=float(data.get("mu", 1.0e-3)))

    def update(self, _time: float) -> None:
        return

    def density(self) -> float:
        return self.rho

    def viscosity(self) -> float:
        return self.mu
