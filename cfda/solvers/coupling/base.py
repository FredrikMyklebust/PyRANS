"""Base pressure-velocity coupling agent."""

from __future__ import annotations

from abc import ABC, abstractmethod

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

    @abstractmethod
    def solve_step(self, case):
        """Perform one outer iteration or time step."""
