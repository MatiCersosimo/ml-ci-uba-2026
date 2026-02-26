"""Utilities for DGP (data-generating process) simulations."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_dgp(
    p: float = 0.5,
    n: int = 1_000,
    n_covariates: int = 3,
    rct: bool = True,
    seed: int | None = None,
    num_perturbed_features: int | None = None,
) -> pd.DataFrame:
    """Simulate data for an RCT or an observational setting.

    DGP (Python equivalent of the R specification):
        X = matrix of uniform(-1, 1), shape (n, n_covariates)
        propensity = 0.1 + 0.85 * sqrt(max(0, (1 + X[:,0] + X[:,1]) / 3))
        If rct: W ~ Bernoulli(p). If not rct: W ~ Bernoulli(propensity)
        Y = W * max(0, X[:,0]) + exp(X[:,1] + X[:,2])
        If num_perturbed_features = p > 0: U_ijp = X_ij + ε_ijp, ε_ijp ~ N(0, 10),
        for j in {1,2,3} (X1,X2,X3) and p perturbations (columns U1_1..U1_p, U2_1..U2_p, U3_1..U3_p).

    When `rct=True`, treatment is randomized with constant probability `p`.
    When `rct=False`, treatment follows the propensity score based on X.

    Parameters
    ----------
    p:
        Treatment probability used only when `rct=True`. Must satisfy
        0 < p < 1. Default is 0.5.
    n:
        Number of rows to simulate. Must be a positive integer. Default is 1000.
    n_covariates:
        Number of covariate columns in X. Must be >= 3 (needed for Y).
        Default is 3.
    rct:
        If True, W ~ Bernoulli(p). If False, W ~ Bernoulli(propensity).
        Default is True.
    seed:
        Random seed for reproducibility. Use None for non-deterministic output.
        Default is None.
    num_perturbed_features:
        If 0 or None, no perturbed features U are generated. If p > 0, generates
        p perturbations of X1, X2, and X3 as U_jk = X_j + ε_jk with ε_jk ~ N(0, 10),
        yielding columns U1_1..U1_p, U2_1..U2_p, U3_1..U3_p. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - X1, X2, ... : covariate columns
        - U1_1, ..., U1_p, U2_1, ..., U3_p : perturbed features (if num_perturbed_features > 0)
        - W: treatment indicator
        - Y: continuous outcome
        - prop: propensity score (only when rct=False)
    """
    if not 0 < p < 1:
        raise ValueError("p must be strictly between 0 and 1.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(n_covariates, int) or n_covariates < 3:
        raise ValueError("n_covariates must be an integer >= 3.")
    if num_perturbed_features is not None and (not isinstance(num_perturbed_features, int) or num_perturbed_features < 0):
        raise ValueError("num_perturbed_features must be None or a non-negative integer.")
    rng = np.random.default_rng(seed)

    # X: (n x n_covariates), uniform on (-1, 1)
    X = rng.uniform(-1.0, 1.0, size=(n, n_covariates))

    # propensity = 0.1 + 0.85 * sqrt(pmax(0, (1 + X[,0] + X[,1])/3))
    # R indices 1-based -> columns 0 and 1 in Python
    propensity = 0.1 + 0.85 * np.sqrt(
        np.maximum(0.0, (1.0 + X[:, 2] + X[:, 1]) / 3.0)
    )
    propensity = np.clip(propensity, 0.0, 1.0)

    if rct:
        w = rng.binomial(1, p, size=n)
    else:
        w = rng.binomial(1, propensity, size=n)

    # Y: RCT -> 0.25 * W + exp(X1+X2); observational -> W * p * pmax(0, X3) + exp(X1+X2)
    if rct:
        treatment_term = 0.25 * w
    else:
        treatment_term = w * np.maximum(0.0, X[:, 2])
    y = treatment_term + np.exp(X[:, 0] + X[:, 1])

    out = {"W": w, "Y": y}
    for j in range(n_covariates):
        out[f"X{j + 1}"] = X[:, j]

    # Perturbed features: U_ijp = X_ij + ε_ijp, ε_ijp ~ N(0, 10), j in {1,2,3}, p perturbations
    if num_perturbed_features and num_perturbed_features > 0:
        eps_std = np.sqrt(1)
        for j in range(3):  # X1, X2, X3
            for pert in range(1, num_perturbed_features + 1):
                eps = rng.normal(0, eps_std, size=n)
                out[f"U{j + 1}_{pert}"] = X[:, j] + eps

    if not rct:
        out["prop"] = propensity
    return pd.DataFrame(out)


def _parse_treatment_var(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]] | None,
) -> tuple[str, int | float, int | float]:
    """Parse treatment_var as in estimators.py; return (treatment_col, treatment_value, control_value)."""
    if treatment_var is None:
        treatment_var = {"W": [1, 0]}
    if len(treatment_var) != 1:
        raise ValueError("treatment_var must have a single key.")
    treatment_col, values = next(iter(treatment_var.items()))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("treatment_var values must be [treatment, control].")
    treatment_value, control_value = values
    if treatment_col not in df.columns:
        raise ValueError(f"treatment column '{treatment_col}' not found in df.")
    return treatment_col, treatment_value, control_value


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean: sum(w*x)/sum(w)."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    sw = np.sum(w)
    return float(np.sum(w * x) / sw) if sw > 0 else float("nan")


def _weighted_var(x: np.ndarray, w: np.ndarray, weighted_mean: float) -> float:
    """Weighted variance: sum(w*(x - mean)^2)/sum(w)."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    sw = np.sum(w)
    if sw <= 0:
        return 0.0
    return float(np.sum(w * (x - weighted_mean) ** 2) / sw)


def balance_check(
    df: pd.DataFrame,
    cov: Sequence[str],
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]] | None = None,
    propensity_col: str | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """Balance check: table of standardized mean differences and optional love plot.

    Parameters
    ----------
    df:
        Input DataFrame with treatment and covariate columns.
    cov:
        List of covariate column names to compare across treatment groups.
    treatment_var:
        Mapping with a single key (treatment column name) and a two-element
        list/tuple [treatment, control]. Defaults to {"W": [1, 0]}.
    propensity_col:
        If provided, balance is computed with IPW weighting: treatment units
        get weight 1/propensity, control units get weight 1/(1 - propensity).
        Column must exist in df and contain the propensity score e(X) = P(W=1|X).
    plot:
        If True, draw the balance (love) plot. Default is True.

    Returns
    -------
    pandas.DataFrame
        Table with covariates as index and columns: mean_1, mean_0, difference,
        standardized_difference. mean_1 is mean in treatment group, mean_0 in control.
    """
    treatment_col, treatment_value, control_value = _parse_treatment_var(df, treatment_var)
    for c in cov:
        if c not in df.columns:
            raise ValueError(f"covariate column '{c}' not found in df.")
    if propensity_col is not None and propensity_col not in df.columns:
        raise ValueError(f"propensity column '{propensity_col}' not found in df.")

    treated = df[df[treatment_col] == treatment_value].copy()
    control = df[df[treatment_col] == control_value].copy()
    if treated.empty or control.empty:
        raise ValueError("Both treatment and control groups must be non-empty.")

    if propensity_col is not None:
        prop_t = treated[propensity_col].astype(float).values
        prop_c = control[propensity_col].astype(float).values
        w1 = 1.0 / np.clip(prop_t, 1e-6, None)
        w0 = 1.0 / np.clip(1.0 - prop_c, 1e-6, None)
    else:
        w1 = np.ones(len(treated))
        w0 = np.ones(len(control))

    rows = []
    for c in cov:
        x1 = treated[c].astype(float).values
        x0 = control[c].astype(float).values
        mean_1 = _weighted_mean(x1, w1)
        mean_0 = _weighted_mean(x0, w0)
        diff = mean_1 - mean_0
        var_1 = _weighted_var(x1, w1, mean_1) if len(x1) > 1 else 0.0
        var_0 = _weighted_var(x0, w0, mean_0) if len(x0) > 1 else 0.0
        pooled_var = var_1 + var_0
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
        std_diff = float(diff / pooled_std) if pooled_std > 0 else 0.0
        rows.append({"mean_1": mean_1, "mean_0": mean_0, "difference": diff, "standardized_difference": std_diff})

    table = pd.DataFrame(rows, index=pd.Index(cov, name="covariate"))

    if plot:
        fig, ax = plt.subplots(figsize=(6, max(4, len(cov) * 0.35)))
        y_pos = np.arange(len(cov))[::-1]
        std_diffs = table["standardized_difference"].values
        ax.scatter(std_diffs, y_pos, color="purple", s=40, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cov)
        ax.set_ylabel("Covariate")
        ax.set_xlabel("Standardized Mean Difference")
        finite = std_diffs[np.isfinite(std_diffs)]
        if len(finite) > 0:
            x_min, x_max = np.min(finite), np.max(finite)
            padding = 0.02 if x_max > x_min else 0.05
            ax.set_xlim(min(-0.25, x_min - padding), max(0.25, x_max + padding))
        else:
            ax.set_xlim(-0.25, 0.25)
        ax.set_xticks([-0.25, -0.10, 0.0, 0.10, 0.25])
        ax.axvline(0, color="black", lw=1, zorder=1)
        ax.axvline(-0.10, color="black", lw=1, ls="--", zorder=1)
        ax.axvline(0.10, color="black", lw=1, ls="--", zorder=1)
        ax.axvline(-0.25, color="black", lw=1, ls=":", zorder=1)
        ax.axvline(0.25, color="black", lw=1, ls=":", zorder=1)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        plt.show()

    return table
