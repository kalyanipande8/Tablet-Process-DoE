"""Run the full Tablet Manufacturability & Roller Compaction DoE pipeline."""
import json
import os

from design_ccd import build_ccd
from model_tensile_strength import fit_and_evaluate
from optimize_ribbon_density import (
    find_optimum,
    fit_quadratic_rsm,
    simulate_ccd_responses,
    variability_reduction,
)
from simulate_blend_data import simulate as simulate_blends
from plot_response_surface import make_plots

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)


def main():
    print("=== Step 1: Simulate powder-blend records ===")
    blends = simulate_blends()
    blends.to_csv("data/blend_records.csv", index=False)
    print(f"Simulated {len(blends)} records -> data/blend_records.csv")

    print("\n=== Step 2: Fit tensile-strength regression model ===")
    model, model_results, _ = fit_and_evaluate(blends)
    print(json.dumps(model_results, indent=2))
    with open("output/model_results.json", "w") as f:
        json.dump(model_results, f, indent=2)

    print("\n=== Step 3: Build 30-run central composite design ===")
    ccd = build_ccd()
    ccd.to_csv("data/ccd_design.csv", index=False)
    print(f"CCD runs: {len(ccd)} "
          f"(factorial={sum(ccd.point_type == 'factorial')}, "
          f"axial={sum(ccd.point_type == 'axial')}, "
          f"center={sum(ccd.point_type == 'center')})")

    print("\n=== Step 4: Simulate ribbon-density response & fit RSM ===")
    ccd_results = simulate_ccd_responses()
    ccd_results.to_csv("data/ccd_results.csv", index=False)
    rsm_model, poly, rsm_r2 = fit_quadratic_rsm(ccd_results)
    print(f"Quadratic RSM R^2: {rsm_r2:.4f}")

    print("\n=== Step 5: Locate optimum & compute variability reduction ===")
    optimum = find_optimum(rsm_model, poly)
    var_result = variability_reduction(optimum)
    print(json.dumps({"optimum": optimum, "variability": var_result}, indent=2))

    summary = {
        "tensile_strength_model": model_results,
        "rsm_r2": rsm_r2,
        "optimum": optimum,
        "variability": var_result,
    }
    with open("output/ccd_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Step 6: Generate figures ===")
    make_plots(blends, ccd_results, rsm_model, poly, optimum)
    print("Figures written to output/")

    print("\n=== Summary ===")
    print(f"Tensile-strength model R^2 (held-out): {model_results['selected_r2']:.3f}")
    print(f"Ribbon-density variability reduction: {var_result['pct_variability_reduction']:.1f}%")


if __name__ == "__main__":
    main()
