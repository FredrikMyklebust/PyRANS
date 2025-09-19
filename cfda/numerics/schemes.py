"""Interpolation schemes."""

from __future__ import annotations

from ..utils.registry import Registry


scheme_registry = Registry("schemes")


class Scheme:
    name = "generic"


@scheme_registry.register("linear")
class LinearScheme(Scheme):
    name = "Linear"


@scheme_registry.register("upwind")
class UpwindScheme(Scheme):
    name = "Upwind"
