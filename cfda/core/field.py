"""Field containers for collocated FV variables."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .mesh import Mesh


class Field:
    """Base class for collocated fields."""

    def __init__(self, name: str, mesh: Mesh, values: Iterable[float]) -> None:
        self.name = name
        self.mesh = mesh
        arr = np.asarray(list(values), dtype=float)
        if arr.shape[0] != mesh.ncells:
            raise ValueError(f"Field {name} expects {mesh.ncells} cells, got {arr.shape[0]}")
        self.values = arr

    def copy(self, name: Optional[str] = None) -> "Field":
        dup = self.__class__(name or self.name, self.mesh, self.values.copy())
        return dup

    def fill(self, value: float) -> None:
        self.values[:] = value

    def norm(self) -> float:
        return float(np.linalg.norm(self.values))

    def __array__(self) -> np.ndarray:
        return self.values

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, value) -> None:
        self.values[idx] = value

    def __iadd__(self, other):
        self.values += np.asarray(other)
        return self

    def __isub__(self, other):
        self.values -= np.asarray(other)
        return self

    def __imul__(self, other):
        self.values *= other
        return self


class ScalarField(Field):
    """Scalar field stored at cell centres."""

    def __init__(self, name: str, mesh: Mesh, values: Iterable[float]):
        super().__init__(name, mesh, values)


class VectorField(Field):
    """Vector field with three components per cell."""

    def __init__(self, name: str, mesh: Mesh, values: Iterable[Iterable[float]]):
        arr = np.asarray(list(values), dtype=float)
        if arr.shape != (mesh.ncells, 3):
            raise ValueError(
                f"VectorField {name} expects shape {(mesh.ncells, 3)}, got {arr.shape}"
            )
        self.name = name
        self.mesh = mesh
        self.values = arr

    @property
    def x(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.values[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.values[:, 2]

    def copy(self, name: Optional[str] = None) -> "VectorField":
        return VectorField(name or self.name, self.mesh, self.values.copy())

    def magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.values, axis=1)
