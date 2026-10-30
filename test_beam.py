"""Tesztek a gerendaelemző modulhoz.

Futtatás (pytest nélkül is működik):
    python -m unittest discover -s tests -v
    # vagy, ha telepítve van a pytest:
    python -m pytest -v
"""

import math
import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from beam_analysis import (
    Beam,
    CircularSection,
    CustomSection,
    DistributedLoad,
    ISection,
    MPa,
    RectangularSection,
    plot_diagrams,
    problem_from_dict,
)


def global_extremes(result, quantity):
    cps = [cp for cp in result.critical_points[quantity] if cp.is_global]
    hi = max(cps, key=lambda c: c.value)
    lo = min(cps, key=lambda c: c.value)
    return hi, lo


class TestSimplySupported(unittest.TestCase):
    def test_uniform_load_full_span(self):
        L, q = 6.0, 5000.0
        beam = Beam(L, "simply_supported").add_distributed_load(q)
        res = beam.solve()
        self.assertAlmostEqual(res.reaction_values["R_A"], q * L / 2, places=6)
        self.assertAlmostEqual(res.reaction_values["R_B"], q * L / 2, places=6)
        m_hi, _ = global_extremes(res, "M")
        self.assertAlmostEqual(m_hi.value, q * L**2 / 8, places=4)
        self.assertAlmostEqual(m_hi.x, L / 2, places=6)
        t_hi, t_lo = global_extremes(res, "T")
        self.assertAlmostEqual(t_hi.value, q * L / 2, places=4)
        self.assertAlmostEqual(t_lo.value, -q * L / 2, places=4)

    def test_central_point_load(self):
        L, P = 4.0, 10_000.0
        res = Beam(L).add_point_load(P, L / 2).solve()
        self.assertAlmostEqual(res.reaction_values["R_A"], P / 2)
        m_hi, _ = global_extremes(res, "M")
        self.assertAlmostEqual(m_hi.value, P * L / 4, places=4)
        self.assertAlmostEqual(m_hi.x, L / 2)

    def test_eccentric_point_load(self):
        L, P, a = 5.0, 8000.0, 1.5
        b = L - a
        res = Beam(L).add_point_load(P, a).solve()
        self.assertAlmostEqual(res.reaction_values["R_A"], P * b / L, places=6)
        self.assertAlmostEqual(res.reaction_values["R_B"], P * a / L, places=6)
        m_hi, _ = global_extremes(res, "M")
        self.assertAlmostEqual(m_hi.value, P * a * b / L, places=4)
        self.assertAlmostEqual(m_hi.x, a)

    def test_triangular_load_irrational_maximum(self):
        L, q0 = 6.0, 3000.0
        res = Beam(L).add_distributed_load(0.0, q0).solve()
        self.assertAlmostEqual(res.reaction_values["R_A"], q0 * L / 6, places=6)
        self.assertAlmostEqual(res.reaction_values["R_B"], q0 * L / 3, places=6)
        m_hi, _ = global_extremes(res, "M")
        self.assertAlmostEqual(m_hi.value, q0 * L**2 / (9 * math.sqrt(3)), places=3)
        self.assertAlmostEqual(m_hi.x, L / math.sqrt(3), places=6)

    def test_partial_distributed_load(self):
        L, q, xs, xe = 8.0, 2000.0, 2.0, 5.0
        res = Beam(L).add_distributed_load(q, x_start=xs, x_end=xe).solve()
        w, xc = q * (xe - xs), 0.5 * (xs + xe)
        self.assertAlmostEqual(res.reaction_values["R_B"], w * xc / L, places=6)
        self.assertAlmostEqual(res.reaction_values["R_A"], w - w * xc / L, places=6)
        m_hi, _ = global_extremes(res, "M")
        self.assertAlmostEqual(res.T(m_hi.x + 1e-9), 0.0, delta=1e-3)

    def test_overhanging_beam(self):
        L, q = 6.0, 1000.0
        res = Beam(L, "simply_supported", supports=(1.0, 5.0)).add_distributed_load(q).solve()
        self.assertAlmostEqual(res.reaction_values["R_A"], 3000.0, places=6)
        self.assertAlmostEqual(res.reaction_values["R_B"], 3000.0, places=6)
        self.assertAlmostEqual(res.M(1.0), -500.0, places=6)
        self.assertAlmostEqual(res.M(3.0), 1500.0, places=6)
        m_hi, m_lo = global_extremes(res, "M")
        self.assertAlmostEqual(m_hi.value, 1500.0, places=4)
        self.assertAlmostEqual(m_lo.value, -500.0, places=4)

    def test_equilibrium_and_dM_dx_equals_T(self):
        beam = Beam(7.0)
        beam.add_point_load(4000, 1.0).add_point_load(-1500, 6.0)
        beam.add_distributed_load(1200, 3000, 2.0, 5.5)
        beam.add_distributed_load(500, 500, 0.0, 7.0)
        res = beam.solve()
        xs, ts, ms = res.sample(points_per_interval=500)
        integral = np.concatenate(([0.0], np.cumsum(0.5 * (ts[1:] + ts[:-1]) * np.diff(xs))))
        scale = np.max(np.abs(ms))
        self.assertLess(np.max(np.abs((ms - ms[0]) - integral)), 1e-4 * scale)
        self.assertAlmostEqual(res.M(0.0), 0.0, places=6)
        self.assertAlmostEqual(res.M(7.0), 0.0, places=6)

    def test_piecewise_symbolic_matches_numeric(self):
        res = Beam(4.0).add_point_load(6000, 1.0).add_distributed_load(1000, x_start=2.0, x_end=4.0).solve()
        import sympy as sp
        from beam_analysis import x as xs

        for val in (0.5, 1.5, 2.5, 3.9):
            self.assertAlmostEqual(float(res.M_piecewise.subs(xs, sp.nsimplify(val))), res.M(val), places=6)
            self.assertAlmostEqual(float(res.T_piecewise.subs(xs, sp.nsimplify(val))), res.T(val), places=6)


class TestCantilever(unittest.TestCase):
    def test_fixed_left_end_load(self):
        L, P = 2.0, 5000.0
        res = Beam(L, "cantilever").add_point_load(P, L).solve()
        self.assertAlmostEqual(res.reaction_values["R"], P)
        self.assertAlmostEqual(res.reaction_values["M_R"], -P * L)
        _, m_lo = global_extremes(res, "M")
        self.assertAlmostEqual(m_lo.value, -P * L, places=4)
        self.assertAlmostEqual(m_lo.x, 0.0)
        self.assertAlmostEqual(res.M(L), 0.0, places=6)

    def test_fixed_left_uniform_load(self):
        L, q = 3.0, 2000.0
        res = Beam(L, "konzolos").add_distributed_load(q).solve()
        self.assertAlmostEqual(res.reaction_values["R"], q * L)
        self.assertAlmostEqual(res.reaction_values["M_R"], -q * L**2 / 2)
        for xv in (0.0, 0.7, 1.5, 2.9):
            self.assertAlmostEqual(res.M(xv), -q * (L - xv) ** 2 / 2, places=6)

    def test_fixed_right_end(self):
        L, P = 2.5, 4000.0
        res = Beam(L, "cantilever", fixed_end="right").add_point_load(P, 0.0).solve()
        self.assertAlmostEqual(res.reaction_values["R"], P)
        self.assertAlmostEqual(res.reaction_values["M_R"], P * L)
        _, m_lo = global_extremes(res, "M")
        self.assertAlmostEqual(m_lo.value, -P * L, places=4)
        self.assertAlmostEqual(m_lo.x, L)

    def test_trapezoid_load(self):
        L, q1, q2 = 2.5, 2000.0, 500.0
        res = Beam(L, "cantilever").add_distributed_load(q1, q2).solve()
        w = (q1 + q2) / 2 * L
        self.assertAlmostEqual(res.reaction_values["R"], w, places=6)
        xc = L * (q1 + 2 * q2) / (3 * (q1 + q2))
        self.assertAlmostEqual(res.reaction_values["M_R"], -w * xc, places=4)


class TestSections(unittest.TestCase):
    def test_rectangle(self):
        s = RectangularSection.from_mm(60, 120)
        self.assertAlmostEqual(s.I, 0.06 * 0.12**3 / 12)
        self.assertAlmostEqual(s.c, 0.06)
        self.assertAlmostEqual(s.W, s.I / s.c)

    def test_i_section_and_circle_and_custom(self):
        s = ISection.from_mm(200, 100, 5.6, 8.5)
        expected = (100 * 200**3 - (100 - 5.6) * (200 - 17) ** 3) / 12 * 1e-12
        self.assertAlmostEqual(s.I, expected, places=12)
        self.assertAlmostEqual(CircularSection(0.1).I, math.pi * 0.1**4 / 64)
        c = CustomSection.from_cm4_mm(1943, 100)
        self.assertAlmostEqual(c.I, 1943e-8)

    def test_invalid_sections(self):
        with self.assertRaises(ValueError):
            RectangularSection(-1, 2)
        with self.assertRaises(ValueError):
            ISection(0.2, 0.1, 0.12, 0.01)
        with self.assertRaises(ValueError):
            ISection(0.2, 0.1, 0.005, 0.1)


class TestStrengthCheck(unittest.TestCase):
    def setUp(self):
        self.res = Beam(4.0).add_point_load(10_000, 2.0).solve()
        self.section = RectangularSection.from_mm(60, 120)

    def test_passes(self):
        chk = self.res.check_strength(self.section, 160 * MPa)
        self.assertAlmostEqual(chk.M_max, 10_000.0, places=4)
        self.assertAlmostEqual(chk.sigma_max / MPa, 10_000 * 0.06 / 8.64e-6 / 1e6, places=4)
        self.assertTrue(chk.ok)
        self.assertLess(chk.utilization, 1.0)
        self.assertGreater(chk.safety_factor, 1.0)

    def test_fails(self):
        chk = self.res.check_strength(self.section, 50 * MPa)
        self.assertFalse(chk.ok)
        self.assertGreater(chk.utilization, 1.0)
        self.assertIn("NEM FELEL MEG", chk.summary())

    def test_uses_absolute_moment_for_cantilever(self):
        res = Beam(2.0, "cantilever").add_point_load(1000, 2.0).solve()
        chk = res.check_strength(self.section, 100 * MPa)
        self.assertAlmostEqual(chk.M_max, -2000.0, places=4)
        self.assertGreater(chk.sigma_max, 0)

    def test_invalid_allowable_stress(self):
        with self.assertRaises(ValueError):
            self.res.check_strength(self.section, 0.0)


class TestValidationAndIO(unittest.TestCase):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            Beam(-1.0)
        with self.assertRaises(ValueError):
            Beam(5.0, "fantázia")
        with self.assertRaises(ValueError):
            Beam(5.0, "simply_supported", supports=(3.0, 1.0))
        with self.assertRaises(ValueError):
            Beam(5.0, "cantilever", supports=(0.0, 5.0))
        with self.assertRaises(ValueError):
            Beam(5.0).add_point_load(100, 6.0)
        with self.assertRaises(ValueError):
            Beam(5.0).add_distributed_load(100, x_start=2.0, x_end=7.0)
        with self.assertRaises(ValueError):
            DistributedLoad(1.0, 1.0, 3.0, 3.0)
        with self.assertRaises(ValueError):
            Beam(5.0).solve()

    def test_problem_from_dict(self):
        data = {
            "length_m": 4.0,
            "type": "kéttámaszú",
            "point_loads": [{"force_N": 10000, "x_m": 2.0}],
            "section": {"type": "rectangular", "b_mm": 60, "h_mm": 120},
            "sigma_allow_MPa": 160,
        }
        problem = problem_from_dict(data)
        res = problem.beam.solve()
        chk = res.check_strength(problem.section, problem.sigma_allow)
        self.assertTrue(chk.ok)
        without_sigma = {k: v for k, v in data.items() if k != "sigma_allow_MPa"}
        with self.assertRaises(ValueError):
            problem_from_dict(without_sigma)

    def test_plot_smoke(self):
        beam = Beam(6.0).add_point_load(12_000, 2.0).add_distributed_load(4000, x_start=3.0, x_end=6.0)
        res = beam.solve()
        chk = res.check_strength(ISection.from_mm(200, 100, 5.6, 8.5), 160 * MPa)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "abra.png")
            fig = plot_diagrams(res, chk, save_path=path, show=False)
            self.assertTrue(os.path.getsize(path) > 10_000)
            self.assertEqual(len(fig.axes), 2)
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_summary_text(self):
        res = Beam(4.0).add_point_load(10_000, 2.0).solve()
        text = res.summary()
        self.assertIn("R_A", text)
        self.assertIn("|M|max", text)


if __name__ == "__main__":
    unittest.main()