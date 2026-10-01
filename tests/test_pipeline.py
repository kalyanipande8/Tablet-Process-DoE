import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_ccd import build_ccd
from model_tensile_strength import fit_and_evaluate
from optimize_ribbon_density import (
    find_optimum,
    fit_quadratic_rsm,
    simulate_ccd_responses,
    variability_reduction,
)
from simulate_blend_data import simulate as simulate_blends


def test_blend_record_count_and_columns():
    df = simulate_blends()
    assert len(df) >= 300
    expected_cols = {
        "particle_size_um",
        "compression_force_kN",
        "lubricant_pct",
        "moisture_pct",
        "relative_density",
        "tensile_strength_mpa",
    }
    assert expected_cols.issubset(df.columns)
    assert (df["tensile_strength_mpa"] > 0).all()


def test_tensile_strength_model_r2_in_reasonable_band():
    df = simulate_blends()
    _, results, _ = fit_and_evaluate(df)
    r2 = results["selected_r2"]
    assert 0.70 <= r2 <= 0.98, f"R^2 {r2} outside reasonable band"


def test_ccd_structure():
    df = build_ccd()
    assert len(df) == 30
    assert sum(df.point_type == "factorial") == 8
    assert sum(df.point_type == "axial") == 6
    assert sum(df.point_type == "center") == 16

    # Factorial points should all be coded +/-1 in every factor
    factorial = df[df.point_type == "factorial"]
    for col in ["coded_roll_pressure_bar", "coded_roll_speed_rpm", "coded_gap_width_mm"]:
        assert set(np.round(factorial[col].unique(), 6)) <= {-1.0, 1.0}

    # Center points should be coded (0,0,0)
    center = df[df.point_type == "center"]
    for col in ["coded_roll_pressure_bar", "coded_roll_speed_rpm", "coded_gap_width_mm"]:
        assert np.allclose(center[col].values, 0.0)

    # Exactly one factor nonzero per axial point, at +/- alpha
    axial = df[df.point_type == "axial"]
    coded_cols = ["coded_roll_pressure_bar", "coded_roll_speed_rpm", "coded_gap_width_mm"]
    nonzero_counts = (axial[coded_cols].abs() > 1e-9).sum(axis=1)
    assert (nonzero_counts == 1).all()


def test_rsm_fit_and_optimum_within_bounds():
    ccd_results = simulate_ccd_responses()
    model, poly, r2 = fit_quadratic_rsm(ccd_results)
    assert r2 > 0.9
    optimum = find_optimum(model, poly)
    assert 30 <= optimum["roll_pressure_bar"] <= 90
    assert 5 <= optimum["roll_speed_rpm"] <= 15
    assert 2 <= optimum["gap_width_mm"] <= 4


def test_variability_reduction_in_reasonable_band():
    ccd_results = simulate_ccd_responses()
    model, poly, _ = fit_quadratic_rsm(ccd_results)
    optimum = find_optimum(model, poly)
    var_result = variability_reduction(optimum)
    pct = var_result["pct_variability_reduction"]
    assert 15 <= pct <= 55, f"Variability reduction {pct} outside reasonable band"
    assert var_result["optimum_std"] < var_result["baseline_std"]
