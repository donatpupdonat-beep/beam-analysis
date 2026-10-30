"""Példa 3: jobb oldalon befogott konzol, háromszög-teher + felfelé ható erő, katalógus-profil.

    python example_3.py            # megjeleníti az ábrát
    python example_3.py --no-show  # csak menti a docs/ mappába
"""

import sys

from beam_analysis import Beam, plot_diagrams
from beam_analysis.sections import CustomSection
MPa = 1e6


def main(show: bool):
    beam = Beam(length=3.2, beam_type="cantilever", fixed_end="right")
    beam.add_distributed_load(q_start=0, q_end=6_000)
    beam.add_point_load(force=5_000, x=0.8)
    beam.add_point_load(force=-2_000, x=2.0)
    section = CustomSection.from_cm4_mm(inertia_cm4=3892, c_mm=120, label="240 mm-es katalógus-profil")
    result = beam.solve()
    check = result.check_strength(section, sigma_allow=235 * MPa / 1.5)

    print(result.summary(check))
    plot_diagrams(result, check, save_path="docs/example_3.png", show=show)


if __name__ == "__main__":
    main("--no-show" not in sys.argv)
