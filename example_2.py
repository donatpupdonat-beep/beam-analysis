"""Példa 2: túlnyúló (konzolos végű) kéttámaszú gerenda, kör keresztmetszet.

    python example_2.py            # megjeleníti az ábrát
    python example_2.py --no-show  # csak menti a docs/ mappába
"""

import sys

from beam_analysis import Beam, plot_diagrams
from beam_analysis.sections import CircularSection
MPa = 1e6


def main(show: bool):
    beam = Beam(length=8.0, beam_type="simply_supported", supports=(1.5, 6.0))
    beam.add_distributed_load(q_start=2_500, q_end=2_500)
    beam.add_point_load(force=6_000, x=0.0)
    beam.add_point_load(force=9_000, x=7.5)
    beam.add_point_load(force=4_000, x=3.5)

    section = CircularSection.from_mm(d_mm=120)
    result = beam.solve()
    check = result.check_strength(section, sigma_allow=120 * MPa)

    print(result.summary(check))
    plot_diagrams(result, check, save_path="docs/example_2.png", show=show)


if __name__ == "__main__":
    main("--no-show" not in sys.argv)
