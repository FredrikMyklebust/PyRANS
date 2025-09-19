"""Simple time control utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimeControl:
    start: float = 0.0
    end: float = 0.0
    dt: float = 1.0
    steady: bool = True

    def __iter__(self) -> Iterator[float]:
        if self.steady:
            yield 0.0
            return
        t = self.start
        while t <= self.end + 1e-12:
            yield t
            t += self.dt

    @classmethod
    def from_dict(cls, data):
        if not data:
            return cls()
        mode = data.get("mode", "steady").lower()
        if mode == "steady":
            return cls(steady=True)
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 1.0)),
            dt=float(data.get("dt", 1.0)),
            steady=False,
        )
