"""
Fit a regression model predicting tensile strength from powder-blend
features and evaluate held-out R^2.

Uses polynomial-feature-expanded Ridge regression and a RandomForest as a
comparison, picking whichever generalizes best on a held-out test split.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

FEATURES = [
    "particle_size_um",
    "compression_force_kN",
    "lubricant_pct",
    "moisture_pct",
]
TARGET = "tensile_strength_mpa"


def fit_and_evaluate(df, seed=7):
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    poly_model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1.0),
    )
    poly_model.fit(X_train, y_train)
    poly_pred = poly_model.predict(X_test)
    poly_r2 = r2_score(y_test, poly_pred)

    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=6, random_state=seed, min_samples_leaf=3
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    if poly_r2 >= rf_r2:
        best_name, best_model, best_r2, best_pred = "polynomial_ridge", poly_model, poly_r2, poly_pred
    else:
        best_name, best_model, best_r2, best_pred = "random_forest", rf_model, rf_r2, rf_pred

    results = {
        "polynomial_ridge_r2": float(poly_r2),
        "random_forest_r2": float(rf_r2),
        "selected_model": best_name,
        "selected_r2": float(best_r2),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return best_model, results, (y_test, best_pred)


if __name__ == "__main__":
    df = pd.read_csv("data/blend_records.csv")
    model, results, _ = fit_and_evaluate(df)
    print(json.dumps(results, indent=2))
    with open("output/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
