"""Példa 4: hosszú kéttámaszú gerenda, trapéz-teher + ellentétes irányú erők, NEM megfelelő keresztmetszet.

    python example_4.py            # megjeleníti az ábrát
    python example_4.py --no-show  # csak menti a docs/ mappába
"""

import sys

from beam_analysis import Beam, plot_diagrams
from beam_analysis.sections import RectangularSection
MPa = 1e6

def main(show: bool):
    beam = Beam(length=10.0, beam_type="simply_supported")
    beam.add_distributed_load(q_start=1_000, q_end=5_000, x_start=1.0, x_end=7.0)
    beam.add_point_load(force=15_000, x=3.0)
    beam.add_point_load(force=-6_000, x=8.5)

    section = RectangularSection.from_mm(b_mm=80, h_mm=160)
    result = beam.solve()
    check = result.check_strength(section, sigma_allow=100 * MPa)

    print(result.summary(check))
    if not check.ok:
        print(f"\n>> Nem felel meg! Legalább {check.required_section_modulus * 1e6:.0f} cm³ keresztmetszeti tényező kell.")
    plot_diagrams(result, check, save_path="docs/example_4.png", show=show)


if __name__ == "__main__":
    main("--no-show" not in sys.argv)
