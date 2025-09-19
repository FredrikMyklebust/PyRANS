"""Structured mesh definitions for the CFD solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Face:
    """Face connecting two cells or a cell and boundary."""

    owner: int
    neighbour: Optional[int]
    area_vector: np.ndarray
    center: np.ndarray
    patch: Optional[str] = None

    @property
    def area(self) -> float:
        return float(np.linalg.norm(self.area_vector))

    @property
    def normal(self) -> np.ndarray:
        mag = self.area
        if mag == 0.0:
            return np.zeros_like(self.area_vector)
        return self.area_vector / mag


class Mesh:
    """Cartesian structured mesh with collocated storage."""

    def __init__(
        self,
        cell_centers: np.ndarray,
        cell_volumes: np.ndarray,
        faces: Sequence[Face],
        cell_neighbors: Sequence[List[int]],
        cell_faces: Sequence[List[int]],
        shape: Tuple[int, int],
        spacing: Tuple[float, float],
        boundary_patches: Dict[str, List[int]],
    ) -> None:
        self.cell_centers = cell_centers
        self.cell_volumes = cell_volumes
        self.faces = list(faces)
        self.cell_neighbors = [list(nbrs) for nbrs in cell_neighbors]
        self.cell_faces = [list(fids) for fids in cell_faces]
        self.shape = shape
        self.spacing = spacing
        self.boundary_patches: Dict[str, List[int]] = {
            name: list(face_ids) for name, face_ids in boundary_patches.items()
        }

    @property
    def ncells(self) -> int:
        return int(len(self.cell_centers))

    @classmethod
    def structured(
        cls,
        nx: int,
        ny: int,
        lengths: Tuple[float, float] = (1.0, 1.0),
        patch_aliases: Optional[Dict[str, str]] = None,
    ) -> "Mesh":
        if nx <= 0 or ny <= 0:
            raise ValueError("Structured mesh requires nx, ny > 0")
        lx, ly = lengths
        dx = lx / nx
        dy = ly / ny
        spacing = (dx, dy)
        centers: List[np.ndarray] = []
        volumes: List[float] = []
        faces: List[Face] = []
        cell_neighbors: List[List[int]] = [[] for _ in range(nx * ny)]
        cell_faces: List[List[int]] = [[] for _ in range(nx * ny)]
        boundary_patches: Dict[str, List[int]] = {"xmin": [], "xmax": [], "ymin": [], "ymax": []}

        def cell_index(i: int, j: int) -> int:
            return j * nx + i

        for j in range(ny):
            yc = (j + 0.5) * dy
            for i in range(nx):
                xc = (i + 0.5) * dx
                centers.append(np.array([xc, yc, 0.0]))
                volumes.append(dx * dy)

        face_id = 0

        # x-faces (vertical faces across y-direction)
        for j in range(ny):
            yc = (j + 0.5) * dy
            for i in range(nx + 1):
                x = i * dx
                owner = cell_index(i - 1, j) if i > 0 else None
                neighbour = cell_index(i, j) if i < nx else None
                if owner is None and neighbour is None:
                    continue
                area_vector = np.array([dy, 0.0, 0.0])
                patch: Optional[str]
                if owner is None:
                    owner = neighbour
                    neighbour = None
                    area_vector = -area_vector
                    patch = "xmax"
                elif neighbour is None:
                    patch = "xmin"
                else:
                    patch = None
                center = np.array([x, yc, 0.0])
                faces.append(
                    Face(owner=owner, neighbour=neighbour, area_vector=area_vector.copy(), center=center, patch=patch)
                )
                cell_faces[owner].append(face_id)
                if neighbour is not None:
                    cell_faces[neighbour].append(face_id)
                    cell_neighbors[owner].append(neighbour)
                    cell_neighbors[neighbour].append(owner)
                else:
                    boundary_patches[patch].append(face_id)
                face_id += 1

        # y-faces (horizontal faces across x-direction)
        for j in range(ny + 1):
            y = j * dy
            for i in range(nx):
                xc = (i + 0.5) * dx
                owner = cell_index(i, j - 1) if j > 0 else None
                neighbour = cell_index(i, j) if j < ny else None
                if owner is None and neighbour is None:
                    continue
                area_vector = np.array([0.0, dx, 0.0])
                patch: Optional[str]
                if owner is None:
                    owner = neighbour
                    neighbour = None
                    area_vector = -area_vector
                    patch = "ymax"
                elif neighbour is None:
                    patch = "ymin"
                else:
                    patch = None
                center = np.array([xc, y, 0.0])
                faces.append(
                    Face(owner=owner, neighbour=neighbour, area_vector=area_vector.copy(), center=center, patch=patch)
                )
                cell_faces[owner].append(face_id)
                if neighbour is not None:
                    cell_faces[neighbour].append(face_id)
                    cell_neighbors[owner].append(neighbour)
                    cell_neighbors[neighbour].append(owner)
                else:
                    boundary_patches[patch].append(face_id)
                face_id += 1

        if patch_aliases:
            for base_name, alias in patch_aliases.items():
                if base_name not in boundary_patches:
                    raise KeyError(f"Unknown base patch '{base_name}'")
                boundary_patches[alias] = boundary_patches.pop(base_name)
                for fid in boundary_patches[alias]:
                    faces[fid].patch = alias

        return cls(
            cell_centers=np.array(centers),
            cell_volumes=np.array(volumes),
            faces=faces,
            cell_neighbors=cell_neighbors,
            cell_faces=cell_faces,
            shape=(nx, ny),
            spacing=spacing,
            boundary_patches=boundary_patches,
        )

    def neighbors(self, cell_id: int) -> Iterable[int]:
        return self.cell_neighbors[cell_id]

    def faces_for_cell(self, cell_id: int) -> Iterable[int]:
        return self.cell_faces[cell_id]

    def patch_faces(self, name: str) -> List[int]:
        return self.boundary_patches[name]

    def patches(self) -> List[str]:
        return list(self.boundary_patches.keys())
