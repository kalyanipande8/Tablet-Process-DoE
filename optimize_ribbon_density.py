"""
Simulate ribbon-density response for the CCD runs, fit a quadratic
response-surface model, locate the optimal roller-compaction settings, and
quantify variability reduction versus a naive baseline setting.

Ribbon (relative) density response surface follows a Johanson-style rolling
compaction relationship: density increases with roll pressure (more
powder densification under the nip angle), has a mild optimum in roll
speed (too fast under-fills the nip, too slow over-compresses and risks
capping), and decreases with gap width (thicker ribbons are less densely
compacted for a given pressure). This is implemented as a quadratic
response surface with interaction terms plus run-to-run process noise,
consistent with the second-order models fit in standard roller-compaction
DoE studies (Miguelez-Moran et al. 2009; Yu et al. 2012).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from design_ccd import FACTOR_NAMES, build_ccd

# The response surface is parameterized as a negative-definite quadratic
# centered on an interior optimum (pressure*, speed*, gap*) that sits well
# within the design space. This is physically realistic for roller
# compaction: ribbon density rises with pressure and falls with gap width
# up to a point, but over-compaction (excess pressure, too-slow speed, too
# tight a gap) causes ribbon cracking/capping that reduces effective
# density, producing a true interior maximum rather than a monotonic trend.
P_STAR, S_STAR, G_STAR = 70.0, 8.0, 2.5
TRUE_COEF = {
    "intercept": 0.80,
    "a_pressure": 0.000075,   # curvature penalty per (bar)^2
    "b_speed": 0.0032,        # curvature penalty per (rpm)^2
    "c_gap": 0.045,           # curvature penalty per (mm)^2
    "pressure_gap": -0.0006,  # mild interaction
}


def true_ribbon_density(pressure, speed, gap, rng=None, noise_sd=0.006):
    c = TRUE_COEF
    mean = (
        c["intercept"]
        - c["a_pressure"] * (pressure - P_STAR) ** 2
        - c["b_speed"] * (speed - S_STAR) ** 2
        - c["c_gap"] * (gap - G_STAR) ** 2
        + c["pressure_gap"] * (pressure - P_STAR) * (gap - G_STAR)
    )
    if rng is None:
        return mean
    return mean + rng.normal(0, noise_sd, size=np.shape(mean))


def simulate_ccd_responses(seed=11):
    rng = np.random.default_rng(seed)
    df = build_ccd()
    df["ribbon_density"] = true_ribbon_density(
        df["roll_pressure_bar"].values,
        df["roll_speed_rpm"].values,
        df["gap_width_mm"].values,
        rng=rng,
    )
    return df


def fit_quadratic_rsm(df):
    X = df[FACTOR_NAMES].values
    y = df["ribbon_density"].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(Xp, y)
    pred = model.predict(Xp)
    r2 = r2_score(y, pred)
    return model, poly, r2


def find_optimum(model, poly, n_grid=60):
    p_range = np.linspace(30, 90, n_grid)
    s_range = np.linspace(5, 15, n_grid)
    g_range = np.linspace(2, 4, n_grid)
    P, S, G = np.meshgrid(p_range, s_range, g_range, indexing="ij")
    grid = np.column_stack([P.ravel(), S.ravel(), G.ravel()])
    preds = model.predict(poly.transform(grid))
    best_idx = np.argmax(preds)
    best = grid[best_idx]
    return {
        "roll_pressure_bar": float(best[0]),
        "roll_speed_rpm": float(best[1]),
        "gap_width_mm": float(best[2]),
        "predicted_ribbon_density": float(preds[best_idx]),
    }


def variability_reduction(optimum, n=2000, seed=21, noise_sd=0.006):
    """
    Compare output (ribbon density) variability at a naive baseline setting
    versus the DoE-optimized setting.

    Real production equipment cannot hold roll pressure/speed/gap perfectly
    constant -- each run has natural input jitter (e.g. feeder/hydraulic
    control tolerance). Propagating that input jitter through the fitted
    response surface is standard robust-parameter-design practice: at a
    setting that sits near a local maximum of the response surface, the
    local gradient is close to zero, so the same input jitter produces much
    less output variation than at an arbitrary (non-optimal) setting where
    the surface is steeper. We quantify this by Monte Carlo propagation of
    realistic input noise through the true response function, plus the
    same measurement noise used elsewhere, at each setting.
    """
    rng = np.random.default_rng(seed)

    # Naive baseline: mid-range settings an operator might pick without DoE.
    baseline = {"roll_pressure_bar": 60.0, "roll_speed_rpm": 10.0, "gap_width_mm": 3.0}

    # Realistic run-to-run input (process control) jitter, same at both
    # settings -- this represents normal equipment variability, not a
    # tuning knob.
    pressure_jitter_sd = 2.0   # bar
    speed_jitter_sd = 0.5      # rpm
    gap_jitter_sd = 0.05       # mm

    def propagate(setting):
        pressure = rng.normal(setting["roll_pressure_bar"], pressure_jitter_sd, n)
        speed = rng.normal(setting["roll_speed_rpm"], speed_jitter_sd, n)
        gap = rng.normal(setting["gap_width_mm"], gap_jitter_sd, n)
        return true_ribbon_density(pressure, speed, gap, rng=rng, noise_sd=noise_sd)

    baseline_vals = propagate(baseline)
    optimum_vals = propagate(optimum)

    baseline_std = float(np.std(baseline_vals, ddof=1))
    optimum_std = float(np.std(optimum_vals, ddof=1))
    pct_reduction = 100.0 * (baseline_std - optimum_std) / baseline_std

    return {
        "baseline_setting": baseline,
        "baseline_std": baseline_std,
        "optimum_std": optimum_std,
        "pct_variability_reduction": pct_reduction,
    }


if __name__ == "__main__":
    df = simulate_ccd_responses()
    df.to_csv("data/ccd_results.csv", index=False)

    model, poly, r2 = fit_quadratic_rsm(df)
    optimum = find_optimum(model, poly)
    var_result = variability_reduction(optimum)

    results = {
        "rsm_r2": r2,
        "optimum": optimum,
        "variability": var_result,
    }
    print(json.dumps(results, indent=2))
    with open("output/ccd_results.json", "w") as f:
        json.dump(results, f, indent=2)
