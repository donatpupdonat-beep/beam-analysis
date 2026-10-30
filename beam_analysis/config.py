"""Feladat betöltése szótárból / JSON-fájlból.

Példa JSON (mértékegységek a kulcsnevekben):

    {
      "length_m": 6.0,
      "type": "simply_supported",
      "supports_m": [0.0, 6.0],
      "point_loads": [{"force_N": 10000, "x_m": 2.0}],
      "distributed_loads": [
        {"q_start_N_per_m": 5000, "q_end_N_per_m": 5000, "x_start_m": 0.0, "x_end_m": 6.0}
      ],
      "section": {"type": "rectangular", "b_mm": 60, "h_mm": 120},
      "sigma_allow_MPa": 160
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .beam import MPa, Beam
from .sections import CircularSection, CrossSection, CustomSection, ISection, RectangularSection


@dataclass
class Problem:
    """Egy teljes feladat: gerenda + (opcionális) keresztmetszet és σ_meg."""

    beam: Beam
    section: Optional[CrossSection] = None
    sigma_allow: Optional[float] = None  # [Pa]


def _require(data: dict, key: str, context: str) -> Any:
    if key not in data:
        raise KeyError(f"Hiányzó kulcs a(z) {context} megadásában: '{key}'")
    return data[key]


def section_from_dict(data: dict) -> CrossSection:
    """Keresztmetszet létrehozása szótárból (méretek mm-ben)."""
    kind = str(_require(data, "type", "keresztmetszet")).strip().lower()
    if kind in ("rectangular", "rectangle", "téglalap", "teglalap"):
        return RectangularSection.from_mm(_require(data, "b_mm", "téglalap"), _require(data, "h_mm", "téglalap"))
    if kind in ("i", "i_beam", "i-beam", "i-tartó", "i-tarto", "isection"):
        return ISection.from_mm(
            _require(data, "h_mm", "I-tartó"),
            _require(data, "b_mm", "I-tartó"),
            _require(data, "tw_mm", "I-tartó"),
            _require(data, "tf_mm", "I-tartó"),
        )
    if kind in ("circular", "circle", "kör", "kor"):
        return CircularSection.from_mm(_require(data, "d_mm", "kör"))
    if kind in ("custom", "egyedi"):
        return CustomSection.from_cm4_mm(
            _require(data, "I_cm4", "egyedi keresztmetszet"),
            _require(data, "c_mm", "egyedi keresztmetszet"),
            data.get("label", "egyedi keresztmetszet"),
        )
    raise ValueError(f"Ismeretlen keresztmetszet-típus: {kind!r}")


def problem_from_dict(data: dict) -> Problem:
    """Feladat összeállítása szótárból."""
    beam = Beam(
        length=_require(data, "length_m", "gerenda"),
        beam_type=data.get("type", "simply_supported"),
        supports=data.get("supports_m"),
        fixed_end=data.get("fixed_end", "left"),
    )
    for p in data.get("point_loads", []):
        beam.add_point_load(_require(p, "force_N", "pontszerű terhelés"), _require(p, "x_m", "pontszerű terhelés"))
    for d in data.get("distributed_loads", []):
        beam.add_distributed_load(
            _require(d, "q_start_N_per_m", "megoszló terhelés"),
            d.get("q_end_N_per_m"),
            d.get("x_start_m", 0.0),
            d.get("x_end_m"),
        )
    section = section_from_dict(data["section"]) if "section" in data else None
    sigma = float(data["sigma_allow_MPa"]) * MPa if "sigma_allow_MPa" in data else None
    if (section is None) != (sigma is None):
        raise ValueError("A szilárdsági ellenőrzéshez a 'section' és a 'sigma_allow_MPa' együtt szükséges.")
    return Problem(beam, section, sigma)


def load_problem(path: "str | Path") -> Problem:
    """Feladat betöltése JSON-fájlból."""
    with open(path, encoding="utf-8") as fh:
        return problem_from_dict(json.load(fh))