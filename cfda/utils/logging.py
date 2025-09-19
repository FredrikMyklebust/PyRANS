"""Simple logging utilities for solver iterations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IterationLogger:
    name: str
    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, iteration: int, residuals: Dict[str, float]) -> None:
        entry = {"iter": iteration, **residuals}
        self.history.append(entry)
        pieces = [f"{self.name} iter {iteration:3d}"]
        for name, value in residuals.items():
            pieces.append(f"{name} residual = {value:.3e}")
        print(" | ".join(pieces), flush=True)
