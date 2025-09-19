"""SIMPLEC coupling agent (wrapper around SIMPLE for MVP)."""

from __future__ import annotations

from .simple import SimpleCoupling
from .base import register_coupling


@register_coupling("simplec")
class SimpleCCoupling(SimpleCoupling):
    pass
