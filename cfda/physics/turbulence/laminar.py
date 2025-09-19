"""Laminar turbulence model (nut = 0)."""

from __future__ import annotations

import numpy as np

from .base import TurbulenceModel, register_turbulence


@register_turbulence("laminar")
class LaminarModel(TurbulenceModel):
    def correct(self, velocity_field) -> None:
        self._nut.values[:] = 0.0
        self._production.values[:] = 0.0
