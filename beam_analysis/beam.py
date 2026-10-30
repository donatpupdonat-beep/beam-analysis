"""Kéttámaszú és konzolos gerendák statikai elemzése.

Előjelkonvenciók
----------------
- x: a bal végtől mért pozíció [m], 0 <= x <= L.
- Terhelés: lefelé pozitív (erő [N], megoszló terhelés [N/m]).
- Támaszerő: felfelé pozitív. Befogási reakciónyomaték (``M_R``):
  óramutató járásával megegyező értelmű = pozitív.
- Nyíróerő T(x): a metszéstől balra eső külső erők eredője (felfelé pozitív).
- Hajlítónyomaték M(x): a gerendát *alul húzó* (sagging) nyomaték pozitív.
  Így dM/dx = T és dT/dx = −q.

Számítási módszer
-----------------
1. A terhelések eredőjéből és nyomatékából az egyensúlyi egyenletekkel
   (ΣF = 0, ΣM = 0) meghatározzuk a támaszreakciókat.
2. A gerendát töréspontokkal (támaszok, terhelések kezdő/záró pontjai)
   szakaszokra bontjuk; minden szakaszon a metszéstől balra eső terhelésekből
   egzakt SymPy-polinomként állítjuk elő T(x)-et és M(x)-et.
3. A szakaszok végpontjaiban, valamint a T = 0 (M szélsőértéke) és q = 0
   (T szélsőértéke) helyeken kiértékelve megkapjuk a kritikus pontokat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import numpy as np
import sympy as sp

from .loads import DistributedLoad, PointLoad, to_rational
from .sections import CrossSection

x = sp.Symbol("x", real=True)
_xi = sp.Symbol("xi", real=True)

MPa = 1.0e6
kN = 1.0e3


class BeamType(Enum):
    """A gerenda alaptípusa."""

    SIMPLY_SUPPORTED = "kéttámaszú"
    CANTILEVER = "konzolos"

    @classmethod
    def parse(cls, value: "BeamType | str") -> "BeamType":
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "simply_supported": cls.SIMPLY_SUPPORTED,
            "simple": cls.SIMPLY_SUPPORTED,
            "ss": cls.SIMPLY_SUPPORTED,
            "kéttámaszú": cls.SIMPLY_SUPPORTED,
            "kettamaszu": cls.SIMPLY_SUPPORTED,
            "két_támaszú": cls.SIMPLY_SUPPORTED,
            "cantilever": cls.CANTILEVER,
            "konzol": cls.CANTILEVER,
            "konzolos": cls.CANTILEVER,
        }
        if key not in aliases:
            raise ValueError(
                f"Ismeretlen gerendatípus: {value!r}. Használható: 'simply_supported' (kéttámaszú), 'cantilever' (konzolos)."
            )
        return aliases[key]


@dataclass(frozen=True)
class Reaction:
    """Egy támaszreakció.

    ``kind == "force"``: értéke [N], felfelé pozitív.
    ``kind == "moment"``: értéke [N·m], óramutató járásával megegyező = pozitív.
    """

    name: str
    x: float
    value: float
    kind: str


@dataclass(frozen=True)
class CriticalPoint:
    """Kritikus (szélső) igénybevételi pont."""

    quantity: str
    x: float
    value: float
    kind: str
    scope: str

    @property
    def is_global(self) -> bool:
        return self.scope == "globális"


@dataclass(frozen=True)
class StrengthCheck:
    """A hajlítófeszültség és a megengedett feszültség összevetése."""

    section: CrossSection
    M_max: float
    x_M_max: float
    sigma_max: float
    sigma_allow: float

    @property
    def ok(self) -> bool:
        return self.sigma_max <= self.sigma_allow

    @property
    def utilization(self) -> float:
        """Kihasználtság σ_max / σ_meg (≤ 1 esetén megfelel)."""
        return self.sigma_max / self.sigma_allow

    @property
    def safety_factor(self) -> float:
        """Biztonsági tényező σ_meg / σ_max."""
        return math.inf if self.sigma_max == 0 else self.sigma_allow / self.sigma_max

    @property
    def required_section_modulus(self) -> float:
        """Szükséges keresztmetszeti tényező K_szüks = M_max / σ_meg [m³]."""
        return abs(self.M_max) / self.sigma_allow

    def summary(self) -> str:
        verdict = "MEGFELEL ✔" if self.ok else "NEM FELEL MEG ✘"
        return (
            f"Keresztmetszet: {self.section.describe()}\n"
            f"  I = {self.section.I * 1e8:.2f} cm⁴,  c = {self.section.c * 1e3:.2f} mm,  "
            f"K = {self.section.W * 1e6:.2f} cm³\n"
            f"  |M|max = {abs(self.M_max) / 1e3:.3f} kNm (x = {self.x_M_max:.3f} m)\n"
            f"  σ_max = |M|max·c/I = {self.sigma_max / MPa:.2f} MPa\n"
            f"  σ_meg = {self.sigma_allow / MPa:.2f} MPa\n"
            f"  Kihasználtság = {self.utilization * 100:.1f} %,  biztonsági tényező n = {self.safety_factor:.2f}\n"
            f"  K_szüks = {self.required_section_modulus * 1e6:.2f} cm³\n"
            f"  ⇒ Szilárdsági ellenőrzés: {verdict}"
        )


def _make_func(expr: sp.Expr) -> Callable[[np.ndarray], np.ndarray]:
    """SymPy-kifejezésből NumPy-kompatibilis (skalár- és tömbbarát) függvény."""
    raw = sp.lambdify(x, expr, modules="numpy")

    def func(xs):
        arr = np.asarray(xs, dtype=float)
        return np.broadcast_to(np.asarray(raw(arr), dtype=float), arr.shape).copy()

    return func


def _real_roots_in(expr: sp.Expr, a: float, b: float, tol: float = 1e-9) -> list[float]:
    """Az expr polinom valós gyökei az (a, b) nyílt intervallumban."""
    expr = sp.expand(expr)
    if expr == 0 or expr.is_number:
        return []
    poly = sp.Poly(expr, x)
    if poly.degree() <= 0:
        return []
    found: list[float] = []
    for root in sp.real_roots(poly):
        value = float(root)
        if a + tol < value < b - tol and not any(abs(value - f) <= tol for f in found):
            found.append(value)
    return sorted(found)


@dataclass
class Interval:
    """A gerenda egy szakasza, ahol T(x) és M(x) egyetlen polinom."""

    a: float
    b: float
    a_exact: sp.Rational
    b_exact: sp.Rational
    T: sp.Expr
    M: sp.Expr
    T_func: Callable = field(repr=False, default=None)
    M_func: Callable = field(repr=False, default=None)

    def __post_init__(self) -> None:
        self.T_func = _make_func(self.T)
        self.M_func = _make_func(self.M)


@dataclass
class AnalysisResult:
    """A statikai elemzés teljes eredménye."""

    beam: "Beam"
    reactions: list[Reaction]
    intervals: list[Interval]
    critical_points: dict[str, list[CriticalPoint]]
    _breakpoints: np.ndarray = field(repr=False, default=None)

    def _piecewise(self, attr: str) -> sp.Piecewise:
        pieces = []
        last = len(self.intervals) - 1
        for i, iv in enumerate(self.intervals):
            upper = (x <= iv.b_exact) if i == last else (x < iv.b_exact)
            pieces.append((getattr(iv, attr), sp.And(x >= iv.a_exact, upper)))
        return sp.Piecewise(*pieces)

    @property
    def T_piecewise(self) -> sp.Piecewise:
        """T(x) szakaszonként értelmezett szimbolikus függvény [N]."""
        return self._piecewise("T")

    @property
    def M_piecewise(self) -> sp.Piecewise:
        """M(x) szakaszonként értelmezett szimbolikus függvény [N·m]."""
        return self._piecewise("M")

    def _evaluate(self, xs, attr: str):
        scalar = np.ndim(xs) == 0
        arr = np.atleast_1d(np.asarray(xs, dtype=float))
        out = np.full(arr.shape, np.nan)
        bp = self._breakpoints
        tol = 1e-12 * max(1.0, bp[-1])
        inside = (arr >= bp[0] - tol) & (arr <= bp[-1] + tol)
        idx = np.clip(np.searchsorted(bp, arr, side="right") - 1, 0, len(self.intervals) - 1)
        for i, iv in enumerate(self.intervals):
            mask = inside & (idx == i)
            if mask.any():
                out[mask] = getattr(iv, attr + "_func")(arr[mask])
        return float(out[0]) if scalar else out

    def T(self, xs):
        """Nyíróerő T(x) [N]. Ugrásnál a jobb oldali (x⁺) értéket adja."""
        return self._evaluate(xs, "T")

    def M(self, xs):
        """Hajlítónyomaték M(x) [N·m]."""
        return self._evaluate(xs, "M")

    def sample(self, points_per_interval: int = 200):
        """Mintavételezés ábrázoláshoz: (x, T, M) tömbök."""
        if points_per_interval < 2:
            raise ValueError("A szakaszonkénti pontszámnak legalább 2-nek kell lennie.")
        xs_all, ts_all, ms_all = [], [], []
        for iv in self.intervals:
            xs = np.linspace(iv.a, iv.b, points_per_interval)
            xs_all.append(xs)
            ts_all.append(iv.T_func(xs))
            ms_all.append(iv.M_func(xs))
        return np.concatenate(xs_all), np.concatenate(ts_all), np.concatenate(ms_all)

    @property
    def reaction_values(self) -> dict[str, float]:
        """Reakciók név szerint (erők [N], nyomaték [N·m])."""
        return {r.name: r.value for r in self.reactions}

    def _global(self, quantity: str, kind: str) -> CriticalPoint:
        for cp in self.critical_points[quantity]:
            if cp.is_global and cp.kind == kind:
                return cp
        for cp in self.critical_points[quantity]:
            if cp.is_global:
                return cp
        raise RuntimeError("Nincs globális kritikus pont.")

    def _abs_max(self, quantity: str) -> CriticalPoint:
        hi, lo = self._global(quantity, "max"), self._global(quantity, "min")
        return hi if abs(hi.value) >= abs(lo.value) else lo

    @property
    def max_shear(self) -> CriticalPoint:
        """A legnagyobb abszolút értékű nyíróerő pontja."""
        return self._abs_max("T")

    @property
    def max_moment(self) -> CriticalPoint:
        """A legnagyobb abszolút értékű hajlítónyomaték pontja."""
        return self._abs_max("M")

    def check_strength(self, section: CrossSection, sigma_allow: float) -> StrengthCheck:
        """σ_max = |M|max · c / I összevetése a megengedett feszültséggel."""
        if not (math.isfinite(sigma_allow) and sigma_allow > 0):
            raise ValueError("A megengedett feszültségnek pozitív számnak kell lennie.")
        cp = self.max_moment
        return StrengthCheck(
            section=section,
            M_max=cp.value,
            x_M_max=cp.x,
            sigma_max=section.bending_stress(cp.value),
            sigma_allow=float(sigma_allow),
        )

    def summary(self, check: Optional[StrengthCheck] = None) -> str:
        beam = self.beam
        bar = "═" * 62
        lines = [bar, " GERENDA STATIKAI ELEMZÉSE", bar]
        lines.append(f"Típus : {beam.beam_type.value}")
        lines.append(f"Hossz : L = {beam.length:.3f} m")
        if beam.beam_type is BeamType.SIMPLY_SUPPORTED:
            lines.append(f"Támaszok : x = {beam.supports[0]:.3f} m és x = {beam.supports[1]:.3f} m")
        else:
            side = "bal" if beam.fixed_end == "left" else "jobb"
            lines.append(f"Befogás : {side} vég (x = {beam.supports[0]:.3f} m)")

        lines.append("\nTerhelések (lefelé pozitív):")
        for i, p in enumerate(beam.point_loads, 1):
            lines.append(f"  [F{i}] F = {p.force / kN:.3f} kN, x = {p.x:.3f} m")
        for i, d in enumerate(beam.distributed_loads, 1):
            lines.append(
                f"  [q{i}] q = {d.q_start / kN:.3f} → {d.q_end / kN:.3f} kN/m, "
                f"x = {d.x_start:.3f} … {d.x_end:.3f} m"
            )

        lines.append("\nTámaszreakciók:")
        for r in self.reactions:
            if r.kind == "force":
                lines.append(f"  {r.name} = {r.value / kN:.3f} kN (x = {r.x:.3f} m, felfelé pozitív)")
            else:
                sense = "óramutató járásával megegyező" if r.value >= 0 else "óramutatóval ellentétes"
                lines.append(f"  {r.name} = {abs(r.value) / kN:.3f} kNm ({sense}, x = {r.x:.3f} m)")

        lines.append("\nSzakaszonkénti igénybevételi függvények (x [m], T [N], M [N·m]):")
        for iv in self.intervals:
            lines.append(f"  {iv.a:.3f} ≤ x ≤ {iv.b:.3f}:")
            lines.append(f"     T(x) = {sp.sstr(sp.N(iv.T, 6))}")
            lines.append(f"     M(x) = {sp.sstr(sp.N(iv.M, 6))}")

        lines.append("\nKritikus pontok:")
        for q, unit, div in (("T", "kN", kN), ("M", "kNm", kN)):
            for cp in self.critical_points[q]:
                lines.append(
                    f"  {q}(x) {cp.scope} {cp.kind}: {cp.value / div:+.3f} {unit} (x = {cp.x:.3f} m)"
                )
        ms, mm = self.max_shear, self.max_moment
        lines.append(f"\n|T|max = {abs(ms.value) / kN:.3f} kN (x = {ms.x:.3f} m)")
        lines.append(f"|M|max = {abs(mm.value) / kN:.3f} kNm (x = {mm.x:.3f} m)")

        if check is not None:
            lines.append("\nSzilárdsági ellenőrzés:")
            lines.append(check.summary())
        lines.append(bar)
        return "\n".join(lines)


class Beam:
    """Kéttámaszú vagy konzolos gerenda."""

    def __init__(
        self,
        length: float,
        beam_type: "BeamType | str" = BeamType.SIMPLY_SUPPORTED,
        supports: Optional[Sequence[float]] = None,
        fixed_end: str = "left",
    ) -> None:
        length = float(length)
        if not (math.isfinite(length) and length > 0):
            raise ValueError("A gerenda hosszának pozitív számnak kell lennie.")
        self.length = length
        self.beam_type = BeamType.parse(beam_type)
        self.point_loads: list[PointLoad] = []
        self.distributed_loads: list[DistributedLoad] = []
        self.fixed_end: Optional[str] = None

        if self.beam_type is BeamType.SIMPLY_SUPPORTED:
            if supports is None:
                supports = (0.0, length)
            if len(supports) != 2:
                raise ValueError("Kéttámaszú gerendához pontosan két támasz-pozíció szükséges.")
            xa, xb = float(supports[0]), float(supports[1])
            if not (0.0 <= xa < xb <= length):
                raise ValueError(
                    f"A támaszoknak 0 ≤ x_A < x_B ≤ L feltételt kell teljesíteniük (kapott: {xa}, {xb}, L={length})."
                )
            self.supports: tuple[float, ...] = (xa, xb)
        else:
            if supports is not None:
                raise ValueError("Konzolos gerendánál a támasz a befogás; használja a 'fixed_end' paramétert.")
            side = str(fixed_end).strip().lower()
            if side in ("left", "bal"):
                self.fixed_end, self.supports = "left", (0.0,)
            elif side in ("right", "jobb"):
                self.fixed_end, self.supports = "right", (length,)
            else:
                raise ValueError("A 'fixed_end' értéke 'left' (bal) vagy 'right' (jobb) lehet.")

    def add_point_load(self, force: float, x: float) -> "Beam":
        """Pontszerű erő felvétele (lefelé pozitív [N]; x a bal végtől [m])."""
        load = PointLoad(force, x)
        if not (0.0 <= load.x <= self.length):
            raise ValueError(f"A pontszerű terhelés pozíciója (x={x}) a gerendán (0…{self.length} m) kívül esik.")
        self.point_loads.append(load)
        return self

    def add_distributed_load(
        self,
        q_start: float,
        q_end: Optional[float] = None,
        x_start: float = 0.0,
        x_end: Optional[float] = None,
    ) -> "Beam":
        """Lineárisan változó megoszló terhelés felvétele [N/m]."""
        load = DistributedLoad(
            q_start,
            q_start if q_end is None else q_end,
            x_start,
            self.length if x_end is None else x_end,
        )
        if load.x_start < 0.0 or load.x_end > self.length:
            raise ValueError(
                f"A megoszló terhelés ({load.x_start}…{load.x_end} m) a gerendán (0…{self.length} m) kívül esik."
            )
        self.distributed_loads.append(load)
        return self

    def solve(self) -> AnalysisResult:
        """Támaszreakciók, T(x), M(x) és a kritikus pontok meghatározása."""
        if not (self.point_loads or self.distributed_loads):
            raise ValueError("Nincs felvett terhelés: adjon meg legalább egy pontszerű vagy megoszló terhelést.")

        length = to_rational(self.length)
        loads_w = [p.resultant_exact() for p in self.point_loads] + [d.resultant_exact() for d in self.distributed_loads]
        loads_m0 = [p.first_moment_exact() for p in self.point_loads] + [
            d.first_moment_exact() for d in self.distributed_loads
        ]
        total_w = sum(loads_w, sp.Integer(0))
        total_m0 = sum(loads_m0, sp.Integer(0))

        forces: list[tuple[sp.Rational, sp.Rational]] = []
        moments: list[tuple[sp.Rational, sp.Rational]] = []
        reactions: list[Reaction] = []

        if self.beam_type is BeamType.SIMPLY_SUPPORTED:
            xa, xb = (to_rational(s) for s in self.supports)
            r_b = (total_m0 - xa * total_w) / (xb - xa)
            r_a = total_w - r_b
            forces += [(xa, r_a), (xb, r_b)]
            reactions += [
                Reaction("R_A", float(xa), float(r_a), "force"),
                Reaction("R_B", float(xb), float(r_b), "force"),
            ]
        else:
            xf = to_rational(self.supports[0])
            r = total_w
            m_r = -(total_m0 - xf * total_w)
            forces.append((xf, r))
            moments.append((xf, m_r))
            reactions += [
                Reaction("R", float(xf), float(r), "force"),
                Reaction("M_R", float(xf), float(m_r), "moment"),
            ]

        for p in self.point_loads:
            forces.append((to_rational(p.x), -to_rational(p.force)))

        bps = {sp.Integer(0), length}
        bps.update(pos for pos, _ in forces)
        bps.update(pos for pos, _ in moments)
        for d in self.distributed_loads:
            bps.update((to_rational(d.x_start), to_rational(d.x_end)))
        ordered = sorted(bps)

        intervals = []
        for a, b in zip(ordered[:-1], ordered[1:]):
            t_expr, m_expr = self._left_resultants(a, b, forces, moments)
            intervals.append(Interval(float(a), float(b), a, b, t_expr, m_expr))

        t_end, m_end = self._left_resultants(length, length, forces, moments)
        if t_end != 0 or sp.expand(m_end) != 0:
            raise ArithmeticError(
                f"Az egyensúlyi ellenőrzés sikertelen (T_vég = {t_end}, M_vég = {sp.expand(m_end)})."
            )

        critical = {
            "T": _classify(_collect_sequence(intervals, "T"), "T"),
            "M": _classify(_collect_sequence(intervals, "M"), "M"),
        }
        breakpoints = np.array([float(v) for v in ordered])
        return AnalysisResult(self, reactions, intervals, critical, breakpoints)

    def _left_resultants(self, a, b, forces, moments):
        """T és M kifejezése az [a, b] szakaszon."""
        t_expr: sp.Expr = sp.Integer(0)
        m_expr: sp.Expr = sp.Integer(0)
        for pos, f_up in forces:
            if pos <= a:
                t_expr += f_up
                m_expr += f_up * (x - pos)
        for pos, c_cw in moments:
            if pos <= a:
                m_expr += c_cw
        for d in self.distributed_loads:
            xs, xe = to_rational(d.x_start), to_rational(d.x_end)
            if xe <= a:
                w = d.resultant_exact()
                t_expr -= w
                m_expr -= w * x - d.first_moment_exact()
            elif xs <= a and xe >= b:
                q1, q2 = to_rational(d.q_start), to_rational(d.q_end)
                q = q1 + (q2 - q1) * (_xi - xs) / (xe - xs)
                t_expr -= sp.integrate(q, (_xi, xs, x))
                m_expr -= sp.integrate(q * (x - _xi), (_xi, xs, x))
        return sp.expand(t_expr), sp.expand(m_expr)


def _collect_sequence(intervals: list[Interval], quantity: str) -> list[tuple[float, float]]:
    seq: list[tuple[float, float]] = []
    for iv in intervals:
        expr = iv.T if quantity == "T" else iv.M
        func = iv.T_func if quantity == "T" else iv.M_func
        derivative = sp.diff(iv.T, x) if quantity == "T" else iv.T
        seq.append((iv.a, float(expr.subs(x, iv.a_exact))))
        for root in _real_roots_in(derivative, iv.a, iv.b):
            seq.append((root, float(func(np.float64(root)))))
        seq.append((iv.b, float(expr.subs(x, iv.b_exact))))

    scale = max([1.0] + [abs(v) for _, v in seq])
    eps = 1e-9 * scale
    unique: list[tuple[float, float]] = []
    for px, pv in seq:
        if unique and abs(px - unique[-1][0]) <= 1e-9 and abs(pv - unique[-1][1]) <= eps:
            continue
        unique.append((px, pv))
    return unique


def _classify(seq: list[tuple[float, float]], quantity: str) -> list[CriticalPoint]:
    scale = max([1.0] + [abs(v) for _, v in seq])
    eps = 1e-9 * scale
    vmax = max(v for _, v in seq)
    vmin = min(v for _, v in seq)
    gmax = next((px, pv) for px, pv in seq if pv >= vmax - eps)
    gmin = next((px, pv) for px, pv in seq if pv <= vmin + eps)

    points = [CriticalPoint(quantity, gmax[0], gmax[1], "max", "globális")]
    if vmax - vmin > eps:
        points.append(CriticalPoint(quantity, gmin[0], gmin[1], "min", "globális"))

    def is_global(px: float, pv: float) -> bool:
        return any(abs(px - p.x) <= 1e-9 and abs(pv - p.value) <= eps for p in points)

    compact: list[tuple[float, float]] = []
    for px, pv in seq:
        if not compact or abs(pv - compact[-1][1]) > eps:
            compact.append((px, pv))
    for i in range(1, len(compact) - 1):
        (_, prev), (px, cur), (_, nxt) = compact[i - 1], compact[i], compact[i + 1]
        if is_global(px, cur):
            continue
        if cur > prev + eps and cur > nxt + eps:
            points.append(CriticalPoint(quantity, px, cur, "max", "lokális"))
        elif cur < prev - eps and cur < nxt - eps:
            points.append(CriticalPoint(quantity, px, cur, "min", "lokális"))
    return points