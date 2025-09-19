"""Boundary condition base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from ..mesh import Mesh


class BoundaryCondition(ABC):
    def __init__(self, name: str, mesh: Mesh, faces: Iterable[int]) -> None:
        self.name = name
        self.mesh = mesh
        self.faces = list(faces)

    @abstractmethod
    def apply_coeffs(self, matrix, rhs: np.ndarray, field) -> None:
        """Modify matrix/rhs for cells adjacent to the boundary."""

    @abstractmethod
    def update_face_values(self, face_values: np.ndarray, field) -> None:
        """Recompute face values consistent with the boundary condition."""

    def is_dirichlet(self) -> bool:
        """Whether the boundary fixes the field value at the face."""

        return False
