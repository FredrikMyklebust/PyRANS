"""Utility registry for runtime selection."""

from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        norm = key.lower()

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if norm in self._items:
                raise ValueError(f"{self.name} registry already has key {key}")
            self._items[norm] = factory
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        try:
            return self._items[key.lower()]
        except KeyError as exc:
            raise KeyError(f"Unknown {self.name} '{key}'") from exc

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        factory = self.get(key)
        return factory(*args, **kwargs)

    def keys(self):
        return list(self._items.keys())
