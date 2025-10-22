"""Třída reprezentující uzel"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Node:
    name: str
    weight: int | None = None

    def __str__(self):
        w = f"[{self.weight}]" if self.weight is not None else ""
        return f"{self.name}{w}"

    __repr__ = __str__
