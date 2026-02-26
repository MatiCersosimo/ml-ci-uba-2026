"""Estimators for causal inference examples."""

from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def diff_in_means(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]]
    | None = None,
    outcome_var: str = "Y",
) -> dict[str, float]:
    """Compute the difference-in-means estimator.

    Parameters
    ----------
    df:
        Input DataFrame containing treatment and outcome columns.
    treatment_var:
        Mapping with a single key (treatment column name) and a two-element
        list/tuple [treatment, control]. Defaults to {"W": [1, 0]}.
    outcome_var:
        Column name for the outcome variable in `df`. Default is "Y".

    Returns
    -------
    dict
        Dictionary with:
        - "estimate": mean(Y | W=treatment) - mean(Y | W=control)
        - "std_error": estimated standard error of tau_hat
        - "ci_95": 95% normal-approximation confidence interval
        - "ci_size_perc": width of the 95% interval as % of |estimate|
        - "p_value": two-sided p-value for H0: tau = 0

    Raises
    ------
    ValueError
        If required columns are missing or a group is empty.
    """

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
    if outcome_var not in df.columns:
        raise ValueError(f"outcome column '{outcome_var}' not found in df.")

    treated = df.loc[df[treatment_col] == treatment_value, outcome_var]
    control = df.loc[df[treatment_col] == control_value, outcome_var]

    if treated.empty or control.empty:
        raise ValueError("Both treatment and control groups must be non-empty.")

    treated_mean = treated.mean()
    control_mean = control.mean()
    estimate = treated_mean - control_mean

    n = len(df)
    n1 = len(treated)
    n0 = len(control)

    var_treated = ((treated - treated_mean) ** 2).sum()
    var_control = ((control - control_mean) ** 2).sum()

    variance = (n / (n0**2)) * var_control + (n / (n1**2)) * var_treated
    se = (variance / n) ** 0.5
    ci_95 = (estimate - 1.96 * se, estimate + 1.96 * se)
    if se == 0:
        p_value = 0.0 if estimate != 0 else 1.0
    else:
        z = abs(estimate / se)
        p_value = math.erfc(z / math.sqrt(2))

    ci_size = ci_95[1] - ci_95[0]
    ci_size_pct = (ci_size / abs(estimate) * 100.0) if estimate != 0 else float("nan")

    return {
        "estimate": float(estimate),
        "std_error": float(se),
        "ci_95": [float(ci_95[0]), float(ci_95[1])],
        "ci_size_perc": float(ci_size_pct),
        "p_value": float(p_value),
    }


def regression_based(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]]
    | None = None,
    outcome_var: str = "Y",
    cov: Sequence[str] | None = None,
) -> dict[str, float]:
    """Estimate treatment effect using linear regression adjustment.

    Parameters
    ----------
    df:
        Input DataFrame containing treatment, outcome, and covariate columns.
    treatment_var:
        Mapping with a single key (treatment column name) and a two-element
        list/tuple [treatment, control]. Defaults to {"W": [1, 0]}.
    outcome_var:
        Column name for the outcome variable in `df`. Default is "Y".
    cov:
        Optional list of column names to use as covariates. Default is None.

    Returns
    -------
    dict
        Dictionary with:
        - "estimate": regression coefficient on treatment
        - "std_error": standard error for treatment coefficient
        - "ci_95": 95% normal-approximation confidence interval
        - "ci_size_perc": width of the 95% interval as % of |estimate|
        - "p_value": two-sided p-value for H0: tau = 0
    """

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
    if outcome_var not in df.columns:
        raise ValueError(f"outcome column '{outcome_var}' not found in df.")

    cov = list(cov) if cov is not None else []
    for col in cov:
        if col not in df.columns:
            raise ValueError(f"covariate column '{col}' not found in df.")

    w = df[treatment_col].replace({control_value: 0, treatment_value: 1})
    if not set(w.unique()).issubset({0, 1}):
        raise ValueError("treatment column must contain only control/treatment values.")

    X = pd.DataFrame({"_treatment_": w.astype(float)})
    if cov:
        X = pd.concat([X, df[cov].astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome_var].astype(float)

    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC1")
    param_index = model.params.index
    params = pd.Series(robust.params, index=param_index)
    bse = pd.Series(robust.bse, index=param_index)
    pvalues = pd.Series(robust.pvalues, index=param_index)
    conf_int = pd.DataFrame(robust.conf_int(), index=param_index)

    estimate = params["_treatment_"]
    se = bse["_treatment_"]
    ci_low, ci_high = conf_int.loc["_treatment_"].tolist()
    p_value = pvalues["_treatment_"]

    ci_size = ci_high - ci_low
    ci_size_pct = (ci_size / abs(estimate) * 100.0) if estimate != 0 else float("nan")

    return {
        "estimate": float(estimate),
        "std_error": float(se),
        "ci_95": [float(ci_low), float(ci_high)],
        "ci_size_perc": float(ci_size_pct),
        "p_value": float(p_value),
    }


def stratified_estimator(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]]
    | None = None,
    outcome_var: str = "Y",
    cov: str | None = None,
    cutoffs: Sequence[float] | None = None,
) -> dict:
    """Estimate treatment effect by stratifying on a single covariate and weighting by stratum size.

    Strata are left-inclusive intervals [a, b). With cutoffs [2, 5, 10] the bins are:
    (-inf, 2), [2, 5), [5, 10), [10, +inf). Within each stratum the difference-in-means
    estimator is used; the final estimate is the weighted average with weights equal to
    the proportion of the population in each bin.

    Parameters
    ----------
    df:
        Input DataFrame containing treatment, outcome, and covariate columns.
    treatment_var:
        Mapping with a single key (treatment column name) and a two-element
        list/tuple [treatment, control]. Defaults to {"W": [1, 0]}.
    outcome_var:
        Column name for the outcome variable in `df`. Default is "Y".
    cov:
        Single column name to stratify on. Must be exactly one covariate.
    cutoffs:
        Ordered cutoff points defining stratum boundaries. For k cutoffs there are
        k+1 bins: (-inf, c1), [c1, c2), ..., [ck, +inf).

    Returns
    -------
    dict
        Dictionary with:
        - "estimate": weighted average of within-stratum difference-in-means
        - "strata_df": DataFrame of per-stratum results (bin_low, bin_high, estimate, weight, count)
    """

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
    if outcome_var not in df.columns:
        raise ValueError(f"outcome column '{outcome_var}' not found in df.")

    if cov is None:
        raise ValueError("stratified_estimator requires a single covariate (cov).")
    if isinstance(cov, (list, tuple)):
        if len(cov) != 1:
            raise ValueError("stratified_estimator accepts only one covariate, not a list.")
        cov = cov[0]
    if cov not in df.columns:
        raise ValueError(f"covariate column '{cov}' not found in df.")

    if cutoffs is None or len(cutoffs) == 0:
        raise ValueError("cutoffs must be a non-empty list of stratum boundaries.")

    cutoffs = sorted([float(c) for c in cutoffs])
    bins_edges = [-np.inf, *cutoffs, np.inf]
    # left-inclusive intervals [a, b)
    df = df.copy()
    df["_stratum_"] = pd.cut(
        df[cov].astype(float),
        bins=bins_edges,
        right=False,
        labels=False,
    )

    n_total = len(df)
    strata_bins = []
    strata_estimates = []
    strata_weights = []
    strata_counts = []

    for s in range(len(bins_edges) - 1):
        low, high = bins_edges[s], bins_edges[s + 1]
        sub = df.loc[df["_stratum_"] == s]
        n_s = len(sub)
        low_out = round(float(low), 12) if np.isfinite(low) else low
        high_out = round(float(high), 12) if np.isfinite(high) else high
        strata_bins.append((low_out, high_out))
        strata_counts.append(n_s)

        if n_s == 0:
            strata_estimates.append(float("nan"))
            strata_weights.append(0.0)
            continue

        n_treated = (sub[treatment_col] == treatment_value).sum()
        n_control = (sub[treatment_col] == control_value).sum()
        if n_treated == 0 or n_control == 0:
            strata_estimates.append(float("nan"))
            strata_weights.append(n_s / n_total)
            continue

        weight_s = n_s / n_total
        strata_weights.append(weight_s)

        result_s = diff_in_means(
            sub,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
        )
        strata_estimates.append(result_s["estimate"])

    # weighted average (skip strata with no units; nan estimates contribute 0 if weight 0)
    total_weight = sum(w for w, e in zip(strata_weights, strata_estimates) if not math.isnan(e))
    if total_weight == 0:
        estimate = float("nan")
    else:
        estimate = sum(
            w * e for w, e in zip(strata_weights, strata_estimates)
            if not math.isnan(e)
        ) / total_weight

    strata_df = pd.DataFrame(
        {
            "bin_low": [b[0] for b in strata_bins],
            "bin_high": [b[1] for b in strata_bins],
            "estimate": [float(x) for x in strata_estimates],
            "weight": [float(w) for w in strata_weights],
            "count": strata_counts,
        }
    )

    return {
        "estimate": float(estimate),
        "strata_df": strata_df,
    }


def ipw_estimator(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]]
    | None = None,
    outcome_var: str = "Y",
    propensity_col: str = "prop",
) -> dict[str, float]:
    """Estimate the average treatment effect by inverse probability weighting (IPW).

    Uses the Horvitz-Thompson formula: per-unit term
        gamma_ipw_i = W_i*Y_i/e_i - (1-W_i)*Y_i/(1-e_i),
    with e_i the propensity score. The estimate is the mean of gamma_ipw; the
    standard error is from the sample variance of gamma_ipw (CLT).

    Parameters
    ----------
    df:
        Input DataFrame containing treatment, outcome, and propensity columns.
    treatment_var:
        Mapping with a single key (treatment column name) and a two-element
        list/tuple [treatment, control]. Defaults to {"W": [1, 0]}.
    outcome_var:
        Column name for the outcome variable in `df`. Default is "Y".
    propensity_col:
        Column name for the propensity score e(X) = P(W=1|X). Default is "prop".

    Returns
    -------
    dict
        Dictionary with:
        - "estimate": IPW estimate of the ATE
        - "std_error": standard error (from sample variance of gamma_ipw)
        - "ci_95": 95% normal-approximation confidence interval
        - "ci_size_perc": width of the 95% interval as % of |estimate|
        - "p_value": two-sided p-value for H0: tau = 0
    """

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
    if outcome_var not in df.columns:
        raise ValueError(f"outcome column '{outcome_var}' not found in df.")
    if propensity_col not in df.columns:
        raise ValueError(f"propensity column '{propensity_col}' not found in df.")

    w = df[treatment_col].replace({control_value: 0, treatment_value: 1})
    if not set(w.values).issubset({0, 1}):
        raise ValueError("treatment column must contain only control/treatment values.")

    w = np.asarray(w, dtype=float)
    y = np.asarray(df[outcome_var], dtype=float)
    e = np.asarray(df[propensity_col], dtype=float)

    # gamma_ipw_i = W_i*Y_i/e_i - (1-W_i)*Y_i/(1-e_i)
    gamma_ipw = (w * y / e) - ((1 - w) * y / (1 - e))
    n = len(gamma_ipw)
    estimate = float(np.mean(gamma_ipw))
    var_gamma = np.var(gamma_ipw, ddof=1)
    se = float(np.sqrt(var_gamma / n)) if n > 1 else 0.0
    ci_95 = (estimate - 1.96 * se, estimate + 1.96 * se)
    if se == 0:
        p_value = 0.0 if estimate != 0 else 1.0
    else:
        z = abs(estimate / se)
        p_value = float(math.erfc(z / math.sqrt(2)))
    ci_size = ci_95[1] - ci_95[0]
    ci_size_pct = (ci_size / abs(estimate) * 100.0) if estimate != 0 else float("nan")

    return {
        "estimate": estimate,
        "std_error": se,
        "ci_95": [float(ci_95[0]), float(ci_95[1])],
        "ci_size_perc": float(ci_size_pct),
        "p_value": p_value,
    }


def lin_estimator(
    df: pd.DataFrame,
    treatment_var: Mapping[str, list[int | float] | tuple[int | float, int | float]]
    | None = None,
    outcome_var: str = "Y",
    cov: Sequence[str] | None = None,
) -> dict[str, float]:
    """Estimate treatment effect using Lin (2013) regression adjustment.

    The specification is:
        Y ~ alpha + tau * W + beta * X + gamma * W * (X - mean(X))

    Parameters are the same as `regression_based`, with `cov` providing the
    list of covariates X to include and interact with the treatment indicator.
    """

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
    if outcome_var not in df.columns:
        raise ValueError(f"outcome column '{outcome_var}' not found in df.")

    cov = list(cov) if cov is not None else []
    if not cov:
        raise ValueError("lin_estimator requires at least one covariate.")
    for col in cov:
        if col not in df.columns:
            raise ValueError(f"covariate column '{col}' not found in df.")

    w = df[treatment_col].replace({control_value: 0, treatment_value: 1})
    if not set(w.unique()).issubset({0, 1}):
        raise ValueError("treatment column must contain only control/treatment values.")

    X = pd.DataFrame({"_treatment_": w.astype(float)})
    cov_df = df[cov].astype(float)
    centered = cov_df - cov_df.mean()
    interaction = centered.mul(w.astype(float), axis=0)
    interaction.columns = [f"{col}_x_treat" for col in cov]

    X = pd.concat([X, cov_df, interaction], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[outcome_var].astype(float)

    model = sm.OLS(y, X).fit()
    robust = model.get_robustcov_results(cov_type="HC1")
    param_index = model.params.index
    params = pd.Series(robust.params, index=param_index)
    bse = pd.Series(robust.bse, index=param_index)
    pvalues = pd.Series(robust.pvalues, index=param_index)
    conf_int = pd.DataFrame(robust.conf_int(), index=param_index)

    estimate = params["_treatment_"]
    se = bse["_treatment_"]
    ci_low, ci_high = conf_int.loc["_treatment_"].tolist()
    p_value = pvalues["_treatment_"]
    ci_size = ci_high - ci_low
    ci_size_pct = (ci_size / abs(estimate) * 100.0) if estimate != 0 else float("nan")

    return {
        "estimate": float(estimate),
        "std_error": float(se),
        "ci_95": [float(ci_low), float(ci_high)],
        "ci_size_perc": float(ci_size_pct),
        "p_value": float(p_value),
    }


def bootstrap(
    df: pd.DataFrame,
    B: int,
    estimator: Callable[..., dict],
    **estimator_kwargs,
) -> dict[str, float]:
    """Bootstrap an estimator: resample with replacement B times and summarize the distribution.

    Parameters
    ----------
    df:
        Input DataFrame (will be resampled with replacement).
    B:
        Number of bootstrap iterations.
    estimator:
        Callable that takes (df, **kwargs) and returns a dict containing "estimate"
        (e.g. diff_in_means, regression_based, stratified_estimator, ipw_estimator, lin_estimator).
    **estimator_kwargs:
        Keyword arguments passed through to the estimator on each run (e.g. outcome_var, cov, cutoffs).

    Returns
    -------
    dict
        - "mean": mean of the B bootstrap estimates
        - "std_error": standard deviation of the B bootstrap estimates
        - "ci_95_normal": 95% CI from normal approximation (mean ± 1.96 * std_error)
        - "ci_95_percentile": 95% CI from 2.5/97.5 percentiles of the bootstrap distribution
    """
    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer.")
    n = len(df)
    estimates: list[float] = []
    for _ in range(B):
        df_b = df.sample(n=n, replace=True, random_state=None)
        out = estimator(df_b, **estimator_kwargs)
        estimates.append(float(out["estimate"]))
    arr = np.array(estimates, dtype=float)
    mean = float(np.nanmean(arr))
    std_error = float(np.nanstd(arr, ddof=0))
    ci_normal = [mean - 1.96 * std_error, mean + 1.96 * std_error]
    valid = arr[np.isfinite(arr)]
    ci_percentile = (
        [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))]
        if len(valid) >= 2
        else [float("nan"), float("nan")]
    )
    return {
        "mean": mean,
        "std_error": std_error,
        "ci_95_normal": ci_normal,
        "ci_95_percentile": ci_percentile,
    }

def plot_propensities(df: pd.DataFrame, true_prop, pred_prop) -> None:
    """Plot density of true and predicted propensity scores.

    Parameters
    ----------
    df : DataFrame (used when true_prop or pred_prop are column names).
    true_prop : str or array-like. If str, column name in df; else values.
    pred_prop : str or array-like. If str, column name in df; else values.
    """
    true_vals = df[true_prop].values if isinstance(true_prop, str) else np.asarray(true_prop)
    pred_vals = df[pred_prop].values if isinstance(pred_prop, str) else np.asarray(pred_prop)
    kde_true = gaussian_kde(true_vals)
    kde_pred = gaussian_kde(pred_vals)
    x_grid = np.linspace(-0.1, 1.1, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, kde_true(x_grid), label="True propensity")
    plt.plot(x_grid, kde_pred(x_grid), label="Predicted propensity")
    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()