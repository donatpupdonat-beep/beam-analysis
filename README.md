# Python Beam Analysis

Python program for static analysis and strength checking of simply supported and cantilever beams.

## Author

* Pup Donat

## 1. Description

This software performs engineering calculations for bent structural beams:

* **Static Analysis:** Computes reactions and diagrams for point loads and uniform/variable (triangular/trapezoidal) distributed loads.
* **Internal Forces:** Calculates analytical shear force T(x) and bending moment M(x) functions and finds global extremes.
* **Graphical Output:** Automatically generates and saves shear and moment diagrams into the `docs/` folder.
* **Strength Verification:** Computes maximum bending stress based on cross-sectional properties (moment of inertia I, distance c) and compares it against the allowable stress, calculating safety factors and utilization.

## 2. Project Structure

```text
python projekt/
├── beam_analysis/            # Core calculation modules
│   ├── __init__.py
│   ├── beam.py               # Beam class and load handling
│   ├── sections.py           # Cross-sections (rectangle, circle, I-beam, custom)
│   └── plotting.py           # Matplotlib plotting module
├── docs/                     # Generated diagram images (.png)
├── example_1.py              # Example 1: Simple supported beam
├── example_2.py              # Example 2: Overhanging beam
├── example_3.py              # Example 3: Cantilever with triangular load
├── example_4.py              # Example 4: Beam strength failure check
├── example_5.py              # Example 5: Cantilever with I-section
└── README.md
