# Data generating process
_Exported on 2/24/2026 at 16:44:44 GMT-3 from Cursor (2.5.22)_

---

**User**

@c:\Users\mati_\Desktop\ml-ci-uba-2026\Class 3 - Unconfoundedness.ipynb:1-21  I need to generate this DGP. Can you write the code to do that in @scratchpad.py ? Do it similar to @c:\Users\mati_\Desktop\ml-ci-uba-2026\Class 3 - Unconfoundedness.ipynb:1-21 

---

**Cursor**


Implementing the DGP from the image in scratchpad.py. Checking the scratchpad and notebook style.



Writing the DGP from the image to scratchpad.py in the same style as the notebook.



Summary of what’s in `scratchpad.py`:

1. **X** – Three covariates, each iid \(U(-1, 1)\), shape `(n, 3)`.
2. **Propensity** – \(e(X_i) = 0.1 + 0.85 \sqrt{\max(0,\, 1 + X_{2i} + X_{3i})/3}\), using `np.maximum(0, ...)` so the sqrt argument is non-negative.
3. **W** – Treatment from Bernoulli with probability `e_X` (so treatment depends on X2 and X3).
4. **Y** – \(Y_i = \exp(X_{1i} + X_{2i}) + \max(0, X_{3i})\, W_i\), so the treatment effect is \(\max(0, X_{3i})\) when \(W_i = 1\).

The script uses the same structure as the notebook: `n = 100_000`, `np.random.seed(42)`, and a pandas `DataFrame` with columns `Y`, `W`, `X1`, `X2`, `X3`. Run it with:

```bash
python scratchpad.py
```

---

**User**

Now I want to estimate my treatment effect using IPW. Can you create a function that does this taking an argument for the propensity score column in df and place it in @scratchpad.py ?

---

**Cursor**


Adding an IPW (Inverse Probability Weighting) function that takes the dataframe and the propensity-score column name.



Summary of what was added to `scratchpad.py`:

**`ipw_ate(df, propensity_col, ...)`**  
- **ATE formula:**  
  \(\hat{\tau}_{\text{IPW}} = \frac{1}{n} \sum_i \left[ \frac{W_i Y_i}{e(X_i)} - \frac{(1-W_i) Y_i}{1 - e(X_i)} \right]\).

- **Arguments:**  
  - `df`: DataFrame with outcome, treatment, and propensity score.  
  - `propensity_col`: Name of the column containing \(e(X) = P(W=1|X)\).  
  - `y_col`: Outcome column (default `"Y"`).  
  - `w_col`: Treatment column (default `"W"`).  
  - `clip_prop`: Optional `(low, high)` to clip propensity scores (default `(1e-6, 1-1e-6)`). Use `None` to skip clipping.

- **Return:** Single float, the IPW estimate of the ATE.

**Example** (your `df` has a propensity column `"prop"`):

```python
tau_hat = ipw_ate(df, "prop")
```

---

**User**

I don't want the clipping. I want the function to fail with some message indicating that overlap does not hold. Also, I want to compute the standard error of IPW using the sample variance of the scores

---

**Cursor**


Updating the function: remove clipping, raise an error when overlap fails, and compute the standard error from the sample variance of the IPW scores.



Summary of changes:

1. **No clipping**  
   The `clip_prop` argument is removed. Propensity scores are used as-is.

2. **Overlap check**  
   If any propensity score is outside (0, 1), the function raises:
   ```text
   ValueError: Overlap does not hold: all propensity scores must be strictly in (0, 1). Found values outside this range.
   ```

3. **Standard error from score variance**  
   The IPW score is  
   \(\psi_i = \frac{W_i Y_i}{e(X_i)} - \frac{(1-W_i) Y_i}{1 - e(X_i)}\).  
   The estimate is \(\hat{\tau} = \frac{1}{n}\sum_i \psi_i\), and the standard error is  
   \(\widehat{\mathrm{SE}} = \widehat{\sigma}_\psi / \sqrt{n}\),  
   with \(\widehat{\sigma}_\psi\) the sample standard deviation of \(\psi_i\) (ddof=1).

4. **Return value**  
   The function now returns `(tau_hat, se)` instead of just `tau_hat`.

**Usage:**
```python
tau_hat, se = ipw_ate(df, "prop")
```

---

**User**

Now I want to generate the propensity-score stratified estimator. I want a function that takes the same inputs as ipw_ate, and a parameter that defines the strata bounds from a linear space. The function should perform the stratified estimation and return the estimate.

Place it in @scratchpad.py 

---

**Cursor**


Adding the stratified estimator and minimal boilerplate so we can call it. Checking the notebook for the stratified formula:



Writing the stratified estimator function:



Summary of what’s in `scratchpad.py`:

**`stratified_ate(df, propensity_col, strata_bounds, y_col="Y", w_col="W")`**

- **Same interface as `ipw_ate`:** `df`, `propensity_col`, `y_col`, `w_col`.
- **`strata_bounds`:**
  - **int** → bins from a linear grid: `np.linspace(0, 1, strata_bounds + 1)` (e.g. `5` → 5 strata with edges `[0, 0.2, 0.4, 0.6, 0.8, 1]`).
  - **array-like** → use that 1d array as the bin edges (length = number of strata + 1).
- **Logic:** Assign each unit to a stratum by propensity score, compute difference-in-means in each stratum, then  
  \(\hat{\tau}_{\text{STRAT}} = \sum_x (n_x/n)\, \hat{\tau}(x)\).
- **Overlap:** If a stratum has only treated or only control, the function raises a `ValueError` (overlap does not hold in that stratum).
- **Return:** Single float, the stratified ATE estimate.

**Example:**
```python
# 5 strata with equally spaced bounds in [0, 1]
tau_strat = stratified_ate(df, "prop", strata_bounds=5)

# or custom bounds
tau_strat = stratified_ate(df, "prop", strata_bounds=np.linspace(0.1, 0.95, 6))
```

---

**User**

For the variance of tau_strat, I want to use a bootstrap procedure. Can you implement a bootstrap loop not just for that but to use it with other estimation functions as well? Check @Class 3 - Unconfoundedness.ipynb for how I am estimating difference-in-means, regression-based estimators, etc.

---

**Cursor**


Implementing a generic bootstrap that resamples with replacement, applies any estimator, and returns the point estimate (from the original sample), bootstrap SE, and percentile CI. Adding it to `scratchpad.py`.



Summary of what was added to `scratchpad.py`:

**`bootstrap_ate(df, estimator, B=1000, seed=None, **estimator_kwargs)`**

- **Resampling:** Draws `B` samples of size `n` from `df` with replacement (same as in Class 3).
- **Point estimate:** One call to `estimator(df, **estimator_kwargs)` on the original data; that value is the reported point estimate.
- **Bootstrap SE:** Sample standard deviation of the `B` estimates (with `ddof=1`).
- **95% CI:** 2.5th and 97.5th percentiles of the bootstrap distribution (percentile method from the notebook).
- **Return:** `(point_estimate, bootstrap_se, (ci_low, ci_high))`.

**Compatible with any estimator** that takes a DataFrame and returns either a single number or a tuple/list whose first element is the estimate:

- **Stratified (no closed-form variance):**
  ```python
  tau_hat, se, (ci_lo, ci_hi) = bootstrap_ate(
      df, stratified_ate, B=1000, propensity_col="prop", strata_bounds=20
  )
  ```
- **Difference-in-means:** `bootstrap_ate(df, difference_in_means, B=1000)` (with `difference_in_means(df, treatment="W", outcome="Y")`).
- **Regression (Lin):** `bootstrap_ate(df, treatment_effect_ols, B=1000, covariates=["X1","X2","X3"])` (with `treatment_effect_ols` from the notebook that returns `(estimate, se)`).
- **IPW:** `bootstrap_ate(df, ipw_ate, B=1000, propensity_col="prop")` (first element of the returned tuple is used).

`_extract_estimate` is a small helper so that estimators that return `(tau_hat, se)` (e.g. `ipw_ate`, `treatment_effect_ols`) still work: only the first element is used in the bootstrap distribution.

---

**User**

Can we add the ci limits constructed from the bootstrap se?

---

**Cursor**


Adding the normal-based 95% CI (point_estimate ± 1.96 * bootstrap SE) and returning both the percentile and normal CIs.



Summary of changes:

**`bootstrap_ate` now returns four values:**

1. **point_estimate** – from the original sample  
2. **se** – bootstrap standard error  
3. **ci_percentile** – 95% CI from the 2.5th and 97.5th percentiles of the bootstrap distribution  
4. **ci_normal** – 95% CI from the bootstrap SE: `(point_estimate - 1.96*se, point_estimate + 1.96*se)`

**Usage:**
```python
tau_hat, se, ci_pct, ci_norm = bootstrap_ate(df, stratified_ate, B=1000, propensity_col="prop", strata_bounds=20)
# ci_pct = (2.5th, 97.5th) percentile
# ci_norm = (tau_hat - 1.96*se, tau_hat + 1.96*se)
```

So you can compare the percentile-based and normal-based 95% CIs as in the notebook.

---

**User**

@c:\Users\mati_\Desktop\fig_CIO_balance_plot_short.png  I want to create this figure to conduct balance checks across groups in my dataset using the estimated standardized mean differences and generate a plot that looks like the one attached. Can you write a function that does this and place it in scratchpad.py?

---

**Cursor**


Adding a function that computes SMDs with the given formula and draws the balance (love) plot. Checking for an existing balance/SMD reference in the notebook:



Adding the balance-plot function: implementing SMD computation and plotting.



Summary of what was added to `scratchpad.py`:

**1. `standardized_mean_difference(df, covariates, group_col="W", group1=1, group0=0, ddof=1)`**  
- Computes the estimated SMD for each covariate using  
  **Ŝ_j = (X̄_{j1} − X̄_{j0}) / √(s²_{j1} + s²_{j0})**  
  with sample means and variances (default `ddof=1`).  
- Returns a `pd.Series` indexed by covariate name.

**2. `plot_balance(df, covariates, group_col="W", group1=1, group0=0, variable_labels=None, xlim=(-0.25, 0.25), ax=None)`**  
- Builds the balance (love) plot:
  - **X-axis:** “Standardized Difference” with range `xlim` (default −0.25 to 0.25) and ticks at −0.25, −0.10, 0, 0.10, 0.25.
  - **Reference lines:** solid at 0, dashed at ±0.10, dotted at ±0.25.
  - **Y-axis:** one row per variable; labels from `variable_labels` or column names.
  - **Points:** one purple dot per covariate at its SMD.
- Returns the matplotlib `Axes` (and creates a figure if `ax` is `None`).

**Example with your DGP:**

```python
from scratchpad import plot_balance

# Balance plot for X1, X2, X3 (treatment W)
plot_balance(
    df,
    covariates=["X1", "X2", "X3"],
    group_col="W",
    variable_labels={"X1": "X1", "X2": "X2", "X3": "X3"},  # or nicer names
)
plt.show()
```

To get a figure like the reference (with custom names on the y-axis), pass a dict for `variable_labels`, e.g. `{"X1": "Account age in days", "X2": "Number of transactions pre-experiment", ...}`.

---

**User**

Now I want to include an option to reweigh observations in the balance plot using propensity scores with an input propensity_col that should be in the df. Please modify the function to allow that.

---

**Cursor**


Adding IPW reweighting to the SMD computation and passing it through to the balance plot.



Restoring the propensity-clip line.



Adding the missing `propensity_clip` parameter to the function signature.



Summary of changes:

**1. `standardized_mean_difference`**
- **`propensity_col=None`** – If provided, SMDs use IPW weights: weight = 1/e(X) in group 1 and 1/(1−e(X)) in group 0. Weighted means and weighted variances are used in the SMD formula.
- **`propensity_clip=(1e-9, 1 - 1e-9)`** – Propensity scores are clipped to this range before inverting, so weights stay finite when e is near 0 or 1.

**2. `plot_balance`**
- **`propensity_col=None`** – Passed through to `standardized_mean_difference`. If set, the plot shows balance in the IPW-reweighted sample.
- When `propensity_col` is set, the plot title is set to **"Balance (IPW reweighted)"**.

**Usage**

```python
# Unweighted balance (raw groups)
plot_balance(df, ["X1", "X2", "X3"], group_col="W")
plt.show()

# Balance after IPW reweighting (propensity column in df)
plot_balance(df, ["X1", "X2", "X3"], group_col="W", propensity_col="prop")
plt.show()
```

So you can compare unweighted vs IPW-weighted balance by toggling `propensity_col`.

