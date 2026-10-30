"""Példa 5: bal oldalon befogott konzol, két részleges megoszló teher (egyik háromszög), I-tartó.

    python example_5.py            # megjeleníti az ábrát
    python example_5.py --no-show  # csak menti a docs/ mappába
"""

import sys

from beam_analysis import Beam, plot_diagrams
from beam_analysis.sections import ISection
MPa = 1e6


def main(show: bool):
    beam = Beam(length=4.0, beam_type="cantilever", fixed_end="left")
    beam.add_distributed_load(q_start=3_000, q_end=3_000, x_start=0.0, x_end=1.5)  # 3 kN/m a befogásnál
    beam.add_distributed_load(q_start=0, q_end=8_000, x_start=2.0, x_end=3.5)  # háromszög: 0 -> 8 kN/m
    beam.add_point_load(force=7_500, x=4.0)  # 7,5 kN a szabad végen

    section = ISection.from_mm(h_mm=300, b_mm=150, tw_mm=7.1, tf_mm=10.7)
    result = beam.solve()
    check = result.check_strength(section, sigma_allow=160 * MPa)

    print(result.summary(check))
    plot_diagrams(result, check, save_path="docs/example_5.png", show=show)


if __name__ == "__main__":
    main("--no-show" not in sys.argv)
