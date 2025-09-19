"""Base pressure-velocity coupling agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

from ...utils.registry import Registry


coupling_registry = Registry("coupling")


def register_coupling(name: str):
    return coupling_registry.register(name)


def make_coupling(name: str, case, config=None):
    return coupling_registry.create(name, case, config or {})


class CouplingAgent(ABC):
    def __init__(self, case, config=None) -> None:
        self.case = case
        self.config = config or {}
        self.profiling = bool(self.config.get("profiling", False))
        self.timings = defaultdict(float)

    def enable_profiling(self, flag: bool = True) -> None:
        self.profiling = flag
        self.timings.clear()

    def reset_timings(self) -> None:
        self.timings.clear()

    def get_timings(self) -> dict[str, float]:
        return dict(self.timings)

    @abstractmethod
    def solve_step(self, case):
        """Perform one outer iteration or time step."""
