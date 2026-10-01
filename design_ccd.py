"""
Construct a face-centered central composite design (CCD) for roller
compaction with 3 factors:

    x1: roll pressure   (bar)
    x2: roll speed      (rpm)
    x3: gap width       (mm)

A CCD combines:
  - a full 2^k factorial (the "cube" points, coded +/-1)
  - 2k axial ("star") points
  - replicated center points (coded 0)

This implementation builds the design in coded units from first principles
(not a lookup table), then maps coded levels to real engineering units for
each factor. For a face-centered CCD, alpha = 1.0, so axial points sit on
the faces of the cube (within the factor ranges) rather than extending
beyond them, which is convenient when factor ranges are already bounded by
equipment limits.

Run count for k=3: 8 factorial + 6 axial + 16 center = 30 runs. The large
number of center-point replicates is deliberate: it gives a robust
estimate of pure error / process noise at the nominal operating point,
which is exactly what we need later to quantify variability reduction.
"""
import itertools

import numpy as np
import pandas as pd

FACTOR_NAMES = ["roll_pressure_bar", "roll_speed_rpm", "gap_width_mm"]
FACTOR_LOW = np.array([30.0, 5.0, 2.0])     # coded -1
FACTOR_HIGH = np.array([90.0, 15.0, 4.0])   # coded +1
ALPHA = 1.0  # face-centered CCD
N_CENTER = 16


def _coded_to_real(coded):
    mid = (FACTOR_HIGH + FACTOR_LOW) / 2
    half_range = (FACTOR_HIGH - FACTOR_LOW) / 2
    return mid + coded * half_range


def build_ccd():
    # 1) Factorial (cube) points: full 2^3 factorial at +/-1
    factorial = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))

    # 2) Axial (star) points: (+/-alpha, 0, 0), (0, +/-alpha, 0), (0, 0, +/-alpha)
    axial = []
    for i in range(3):
        for sign in (-1.0, 1.0):
            point = [0.0, 0.0, 0.0]
            point[i] = sign * ALPHA
            axial.append(point)
    axial = np.array(axial)

    # 3) Center points: replicated at coded (0, 0, 0)
    center = np.zeros((N_CENTER, 3))

    coded = np.vstack([factorial, axial, center])
    point_type = (
        ["factorial"] * len(factorial)
        + ["axial"] * len(axial)
        + ["center"] * len(center)
    )

    real_units = _coded_to_real(coded)

    df = pd.DataFrame(real_units, columns=FACTOR_NAMES)
    for i, name in enumerate(FACTOR_NAMES):
        df[f"coded_{name}"] = coded[:, i]
    df["point_type"] = point_type
    df.insert(0, "run", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = build_ccd()
    print(f"CCD runs: {len(df)}  (factorial={sum(df.point_type=='factorial')}, "
          f"axial={sum(df.point_type=='axial')}, center={sum(df.point_type=='center')})")
    df.to_csv("data/ccd_design.csv", index=False)
