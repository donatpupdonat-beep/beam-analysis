"""Terhelések: pontszerű erő és lineárisan változó megoszló terhelés.

Mértékegységek (SI alapegységek): erő [N], hosszúság / pozíció [m],
megoszló terhelés intenzitása [N/m].

Előjelkonvenció: a terhelés **lefelé pozitív**. A pozíciókat a gerenda bal
végétől (x = 0) mérjük.

A belső számításokhoz a terhelések egzakt (racionális) értékeit is
szolgáltatják, így a szimbolikus levezetés mentes a lebegőpontos zajtól.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp


def to_rational(value) -> sp.Rational:
    """Szám átalakítása egzakt racionálissá (pl. 0.1 -> 1/10).

    A Python ``repr`` a legrövidebb, még egyértelmű tizedes alakot adja, így
    a felhasználó által beírt értékek pontosan megmaradnak.
    """
    if isinstance(value, sp.Rational):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Érvénytelen (nem véges) szám: {value!r}")
    return sp.Rational(repr(number))


def _require_finite(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"A(z) '{name}' értéke nem véges szám: {value!r}")


@dataclass(frozen=True)
class PointLoad:
    """Pontszerű (koncentrált) erő.

    Attributes:
        force: az erő nagysága [N], lefelé pozitív.
        x: a támadáspont pozíciója a bal végtől [m].
    """

    force: float
    x: float

    def __post_init__(self) -> None:
        _require_finite(force=self.force, x=self.x)

    @property
    def resultant(self) -> float:
        """Eredő erő [N]."""
        return float(self.force)

    def resultant_exact(self) -> sp.Rational:
        return to_rational(self.force)

    def first_moment_exact(self) -> sp.Rational:
        """Az erő nyomatéka az x = 0 pontra (előjeles, W · x)."""
        return to_rational(self.force) * to_rational(self.x)


@dataclass(frozen=True)
class DistributedLoad:
    """Lineárisan változó megoszló terhelés az [x_start, x_end] szakaszon.

    Egyenletes terheléshez ``q_start == q_end``; háromszög- vagy
    trapéz-terheléshez eltérő végértékek adhatók meg.

    Attributes:
        q_start: intenzitás a szakasz elején [N/m], lefelé pozitív.
        q_end: intenzitás a szakasz végén [N/m], lefelé pozitív.
        x_start: a szakasz kezdő pozíciója [m].
        x_end: a szakasz záró pozíciója [m].
    """

    q_start: float
    q_end: float
    x_start: float
    x_end: float

    def __post_init__(self) -> None:
        _require_finite(
            q_start=self.q_start,
            q_end=self.q_end,
            x_start=self.x_start,
            x_end=self.x_end,
        )
        if self.x_end <= self.x_start:
            raise ValueError(
                "A megoszló terhelés záró pozíciójának nagyobbnak kell lennie "
                f"a kezdőnél (x_start={self.x_start}, x_end={self.x_end})."
            )

    @classmethod
    def uniform(cls, q: float, x_start: float, x_end: float) -> "DistributedLoad":
        """Egyenletesen megoszló terhelés."""
        return cls(q, q, x_start, x_end)

    @property
    def length(self) -> float:
        return float(self.x_end - self.x_start)

    def q_at(self, xi: float) -> float:
        """A terhelés intenzitása az xi pontban (a szakaszon kívül 0)."""
        if xi < self.x_start or xi > self.x_end:
            return 0.0
        t = (xi - self.x_start) / (self.x_end - self.x_start)
        return float(self.q_start + (self.q_end - self.q_start) * t)

    def resultant_exact(self) -> sp.Rational:
        """Az eredő erő: a terhelési ábra területe [N]."""
        q1, q2 = to_rational(self.q_start), to_rational(self.q_end)
        length = to_rational(self.x_end) - to_rational(self.x_start)
        return (q1 + q2) * length / 2

    def first_moment_exact(self) -> sp.Rational:
        """Az eloszlás nyomatéka az x = 0 pontra: ∫ q(ξ)·ξ dξ [N·m].

        Zárt alak: x_s·W + l²·(q1 + 2·q2) / 6, ahol l a szakasz hossza.
        """
        q1, q2 = to_rational(self.q_start), to_rational(self.q_end)
        xs, xe = to_rational(self.x_start), to_rational(self.x_end)
        length = xe - xs
        return xs * self.resultant_exact() + length**2 * (q1 + 2 * q2) / 6

    @property
    def resultant(self) -> float:
        return float(self.resultant_exact())

    @property
    def centroid(self) -> float | None:
        """Az eredő támadáspontja (None, ha az eredő nulla)."""
        w = self.resultant_exact()
        if w == 0:
            return None
        return float(self.first_moment_exact() / w)