"""Generate figures: tabletability curve, parity plot, and response surface."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from design_ccd import FACTOR_NAMES


def make_plots(blends, ccd_results, rsm_model, poly, optimum):
    # Figure 1: tabletability (tensile strength vs compaction/relative density)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(
        blends["relative_density"],
        blends["tensile_strength_mpa"],
        c=blends["lubricant_pct"],
        cmap="viridis",
        s=18,
        alpha=0.8,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Lubricant (%)")
    ax.set_xlabel("Relative density")
    ax.set_ylabel("Tensile strength (MPa)")
    ax.set_title("Tabletability: tensile strength vs. relative density")
    fig.tight_layout()
    fig.savefig("output/tabletability_curve.png", dpi=150)
    plt.close(fig)

    # Figure 2: response surface (pressure x gap) at optimum speed
    p_range = np.linspace(30, 90, 60)
    g_range = np.linspace(2, 4, 60)
    P, G = np.meshgrid(p_range, g_range)
    S = np.full_like(P, optimum["roll_speed_rpm"])
    grid = np.column_stack([P.ravel(), S.ravel(), G.ravel()])
    pred = rsm_model.predict(poly.transform(grid)).reshape(P.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(P, G, pred, levels=20, cmap="plasma")
    fig.colorbar(cs, ax=ax, label="Predicted ribbon relative density")
    ax.scatter(
        optimum["roll_pressure_bar"], optimum["gap_width_mm"],
        color="white", edgecolor="black", s=90, marker="*", zorder=5,
        label="DoE-optimized setting",
    )
    ax.scatter(60.0, 3.0, color="cyan", edgecolor="black", s=70, marker="o",
               zorder=5, label="Naive baseline setting")
    ax.set_xlabel("Roll pressure (bar)")
    ax.set_ylabel("Gap width (mm)")
    ax.set_title(f"Ribbon-density response surface (speed={optimum['roll_speed_rpm']:.1f} rpm)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("output/response_surface.png", dpi=150)
    plt.close(fig)
