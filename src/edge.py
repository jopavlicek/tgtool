"""Třída reprezentující hranu"""

from dataclasses import dataclass
from .node import Node

@dataclass(frozen=True)
class Edge:
    source: Node
    target: Node
    weight: int | None = None
    name: str | None = None

    def __str__(self):
        w = f" [{self.weight}]" if self.weight is not None else ""
        n = f"{self.name}: " if self.name else ""
        return f"({n}{self.source} > {self.target}{w})"

    __repr__ = __str__
