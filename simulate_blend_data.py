"""
Simulate powder-blend records and tensile strength outcomes.

Tensile strength model follows the Ryshkewitch-Zapas relationship, which
relates tablet tensile strength to relative density (solid fraction):

    sigma = sigma_0 * exp(-k * (1 - rho_r))

where rho_r is the relative density (porosity-dependent) of the compact,
sigma_0 is the tensile strength at zero porosity, and k is a material-
dependent bonding constant. Relative density itself is driven by
compression force via a Kawakita-style densification relationship:

    rho_r = rho_0 + (1 - rho_0) * (C * P) / (1 + C * P)

where P is compression force and C is a Kawakita compressibility constant.
Particle size, lubricant %, and moisture content modulate rho_0, C, and
sigma_0, as documented in the pharmaceutical compaction literature
(Ryshkewitch 1953; Kawakita & Ludde 1971; Sun & Grant 2001).
"""
import numpy as np
import pandas as pd

N_RECORDS = 320
SEED = 42


def simulate(n=N_RECORDS, seed=SEED):
    rng = np.random.default_rng(seed)

    particle_size_um = rng.uniform(20, 250, n)          # D50, microns
    compression_force_kN = rng.uniform(5, 35, n)         # kN
    lubricant_pct = rng.uniform(0.25, 2.0, n)            # % w/w magnesium stearate
    moisture_pct = rng.uniform(1.0, 5.0, n)              # % w/w

    # Kawakita compressibility constant C: decreases with particle size and
    # lubricant (lubricant impedes densification), increases with moisture
    # (plasticizing effect up to a point).
    C = (
        0.14
        - 0.00028 * particle_size_um
        - 0.045 * lubricant_pct
        + 0.012 * moisture_pct
    )
    C = np.clip(C, 0.01, None)

    rho_0 = 0.45 + 0.0006 * (250 - particle_size_um) / 250 * 0.05  # loose packing density baseline
    rho_0 = np.clip(rho_0, 0.40, 0.55)

    relative_density = rho_0 + (1 - rho_0) * (C * compression_force_kN) / (
        1 + C * compression_force_kN
    )
    relative_density = np.clip(relative_density, 0.5, 0.97)

    # sigma_0 (zero-porosity tensile strength) reduced by excess lubricant
    # (lubricant film weakens interparticle bonding) and fine particle size
    # increases true contact area/bonding.
    sigma_0 = (
        4.2
        - 0.95 * lubricant_pct
        - 0.006 * particle_size_um
        - 0.03 * moisture_pct
    )
    sigma_0 = np.clip(sigma_0, 1.0, None)

    k_bond = 6.5 + 0.02 * particle_size_um - 0.3 * moisture_pct
    k_bond = np.clip(k_bond, 3.0, 12.0)

    tensile_strength_mpa = sigma_0 * np.exp(-k_bond * (1 - relative_density))

    noise = rng.normal(0, 0.135, n)
    tensile_strength_mpa = np.clip(tensile_strength_mpa + noise, 0.05, None)

    df = pd.DataFrame(
        {
            "particle_size_um": particle_size_um,
            "compression_force_kN": compression_force_kN,
            "lubricant_pct": lubricant_pct,
            "moisture_pct": moisture_pct,
            "relative_density": relative_density,
            "tensile_strength_mpa": tensile_strength_mpa,
        }
    )
    return df


if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/blend_records.csv", index=False)
    print(f"Simulated {len(df)} powder-blend records -> data/blend_records.csv")
