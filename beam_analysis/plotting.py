from __future__ import annotations

from typing import Optional

import numpy as np

from .beam import AnalysisResult, BeamType, CriticalPoint, StrengthCheck

_COLOR_T = "#1f4e79"
_COLOR_M = "#7a1f1f"
_COLOR_GLOBAL = "#d62728"
_COLOR_LOCAL = "#ff9f1c"


def _label(cp: CriticalPoint) -> str:
    if cp.is_global:
        name = f"{cp.quantity}_{cp.kind}"
    else:
        name = f"lok. {cp.kind}"
    unit = "kN" if cp.quantity == "T" else "kNm"
    return f"{name} = {cp.value / 1e3:+.2f} {unit}\nx = {cp.x:.2f} m"


def _annotate(ax, points: list[CriticalPoint], length: float, invert: bool) -> None:
    scale = max([1.0] + [abs(p.value) for p in points])
    shown: list[tuple[float, float]] = []
    for cp in points:
        if abs(cp.value) < 1e-9 * scale:
            continue
        if any(abs(cp.x - sx) < 1e-9 and abs(cp.value - sv) < 1e-9 * scale for sx, sv in shown):
            continue
        shown.append((cp.x, cp.value))

        color = _COLOR_GLOBAL if cp.is_global else _COLOR_LOCAL
        ax.plot(
            cp.x,
            cp.value / 1e3,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="black",
            zorder=6,
        )
        visual_up = (cp.value >= 0) != invert
        dy = 32 if visual_up else -32
        dx = 40 if cp.x < 0.5 * length else -40
        ax.annotate(
            _label(cp),
            xy=(cp.x, cp.value / 1e3),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.95),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
            zorder=7,
        )


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=2)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.7, alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _set_ylim(ax, values: np.ndarray, invert: bool) -> None:
    lo, hi = float(np.min(values)), float(np.max(values))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.45 * span, hi + 0.45 * span)
    if invert:
        ax.invert_yaxis()


def plot_diagrams(
    result: AnalysisResult,
    check: Optional[StrengthCheck] = None,
    *,
    save_path: Optional[str] = None,
    show: bool = True,
    invert_moment_axis: bool = True,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (11.0, 8.5),
    dpi: int = 150,
):
    import matplotlib.pyplot as plt

    beam = result.beam
    length = beam.length
    xs, ts, ms = result.sample(points_per_interval=300)
    ts_kn, ms_knm = ts / 1e3, ms / 1e3

    x_t = np.concatenate(([0.0], xs, [length]))
    t_t = np.concatenate(([0.0], ts_kn, [0.0]))

    fig, (ax_t, ax_m) = plt.subplots(2, 1, sharex=True, figsize=figsize, constrained_layout=True)

    ax_t.plot(x_t, t_t, color=_COLOR_T, linewidth=2.0, zorder=4)
    ax_t.fill_between(x_t, 0.0, t_t, color=_COLOR_T, alpha=0.18, zorder=3)
    _style(ax_t, "Nyíróerő-ábra  T(x)", "T(x)  [kN]")
    _set_ylim(ax_t, t_t, invert=False)
    _annotate(ax_t, result.critical_points["T"], length, invert=False)

    ax_m.plot(xs, ms_knm, color=_COLOR_M, linewidth=2.0, zorder=4)
    ax_m.fill_between(xs, 0.0, ms_knm, color=_COLOR_M, alpha=0.18, zorder=3)
    m_title = "Hajlítónyomatéki ábra  M(x)"
    if invert_moment_axis:
        m_title += "  (a húzott oldalra rajzolva)"
    _style(ax_m, m_title, "M(x)  [kNm]")
    ax_m.set_xlabel("x  [m]", fontsize=11)
    _set_ylim(ax_m, ms_knm, invert=invert_moment_axis)
    _annotate(ax_m, result.critical_points["M"], length, invert=invert_moment_axis)

    marker = "^" if beam.beam_type is BeamType.SIMPLY_SUPPORTED else "s"
    for sx in beam.supports:
        for ax in (ax_t, ax_m):
            ax.plot(sx, 0.0, marker=marker, markersize=9, color="black", markerfacecolor="white", zorder=8, clip_on=False)

    for p in beam.point_loads:
        for ax in (ax_t, ax_m):
            ax.axvline(p.x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

    ax_m.set_xlim(-0.04 * length, 1.04 * length)

    if title is None:
        title = f"Gerenda igénybevételi ábrái — {beam.beam_type.value} gerenda, L = {length:.2f} m"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if check is not None:
        verdict = "MEGFELEL" if check.ok else "NEM FELEL MEG"
        relation = "≤" if check.ok else ">"
        text = (
            f"Szilárdsági ellenőrzés ({check.section.describe()}):  "
            f"σ_max = {check.sigma_max / 1e6:.1f} MPa {relation} σ_meg = {check.sigma_allow / 1e6:.1f} MPa  "
            f"→ {verdict}  (kihasználtság: {check.utilization * 100:.0f} %)"
        )
        fig.text(
            0.5,
            -0.015,
            text,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color="#1b7f3b" if check.ok else "#b00020",
        )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig