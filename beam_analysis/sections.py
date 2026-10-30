from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"A(z) '{name}' méretnek pozitív számnak kell lennie.")


class CrossSection(ABC):
    @property
    @abstractmethod
    def I(self) -> float:
        pass

    @property
    @abstractmethod
    def c(self) -> float:
        pass

    @property
    def W(self) -> float:
        return self.I / self.c

    @abstractmethod
    def describe(self) -> str:
        pass

    def bending_stress(self, moment: float) -> float:
        return abs(moment) * self.c / self.I


@dataclass(frozen=True)
class RectangularSection(CrossSection):
    b: float
    h: float

    def __post_init__(self) -> None:
        _positive(b=self.b, h=self.h)

    @classmethod
    def from_mm(cls, b_mm: float, h_mm: float) -> "RectangularSection":
        return cls(b_mm / 1000.0, h_mm / 1000.0)

    @property
    def I(self) -> float:
        return self.b * self.h**3 / 12.0

    @property
    def c(self) -> float:
        return self.h / 2.0

    def describe(self) -> str:
        return f"téglalap {self.b * 1e3:.1f} × {self.h * 1e3:.1f} mm"


@dataclass(frozen=True)
class ISection(CrossSection):
    h: float
    b: float
    tw: float
    tf: float

    def __post_init__(self) -> None:
        _positive(h=self.h, b=self.b, tw=self.tw, tf=self.tf)
        if self.tw >= self.b:
            raise ValueError("A gerinc vastagsága kisebb kell legyen az öv szélességénél.")
        if 2 * self.tf >= self.h:
            raise ValueError("Az övek összvastagsága kisebb kell legyen a magasságnál.")

    @classmethod
    def from_mm(cls, h_mm: float, b_mm: float, tw_mm: float, tf_mm: float) -> "ISection":
        return cls(h_mm / 1000.0, b_mm / 1000.0, tw_mm / 1000.0, tf_mm / 1000.0)

    @property
    def I(self) -> float:
        inner_h = self.h - 2 * self.tf
        return (self.b * self.h**3 - (self.b - self.tw) * inner_h**3) / 12.0

    @property
    def c(self) -> float:
        return self.h / 2.0

    def describe(self) -> str:
        return (
            f"I-tartó h={self.h * 1e3:.1f}, b={self.b * 1e3:.1f}, "
            f"tw={self.tw * 1e3:.1f}, tf={self.tf * 1e3:.1f} mm"
        )


@dataclass(frozen=True)
class CircularSection(CrossSection):
    d: float

    def __post_init__(self) -> None:
        _positive(d=self.d)

    @classmethod
    def from_mm(cls, d_mm: float) -> "CircularSection":
        return cls(d_mm / 1000.0)

    @property
    def I(self) -> float:
        return math.pi * self.d**4 / 64.0

    @property
    def c(self) -> float:
        return self.d / 2.0

    def describe(self) -> str:
        return f"kör Ø{self.d * 1e3:.1f} mm"


@dataclass(frozen=True)
class CustomSection(CrossSection):
    inertia: float
    c_max: float
    label: str = "egyedi keresztmetszet"

    def __post_init__(self) -> None:
        _positive(inertia=self.inertia, c_max=self.c_max)

    @classmethod
    def from_cm4_mm(cls, inertia_cm4: float, c_mm: float, label: str = "egyedi keresztmetszet") -> "CustomSection":
        return cls(inertia_cm4 * 1e-8, c_mm / 1000.0, label)

    @property
    def I(self) -> float:
        return self.inertia

    @property
    def c(self) -> float:
        return self.c_max

    def describe(self) -> str:
        return f"{self.label} (I = {self.inertia * 1e8:.2f} cm⁴, c = {self.c_max * 1e3:.1f} mm)"