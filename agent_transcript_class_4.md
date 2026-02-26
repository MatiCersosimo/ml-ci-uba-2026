# Linear model estimation with scikit-learn
_Exported on 2/26/2026 at 15:56:15 GMT-3 from Cursor (2.5.25)_

---

**User**

I want to estimate a linear model of W on X given my DGP@Class 4 - ML for TE I.ipynb (1-24) . Can we use scikit-learn to do this? A two-liner should suffice. Place it in @scratchpad.py 

---

**Cursor**

Checking the scratchpad and notebook context.


Adding a two-liner that fits a linear model for binary W on X using scikit-learn's LogisticRegression (standard for binary outcomes):


Done. I added a two-liner in `scratchpad.py` that fits **logistic regression** (scikit-learn’s usual linear model for binary targets) of **W** on **X1, X2, X3**.

Run the DGP cell in the notebook so `df` exists, then run the scratchpad (e.g. `%run scratchpad.py` from the notebook, or run the notebook up to and including the DGP cell and then execute `scratchpad.py`). After that, `model.predict_proba(df[["X1","X2","X3"]])[:, 1]` gives estimated propensity scores \(\hat{e}(X)\).

---

**User**

Can you install scikit-learn in the venv of this folder?

---

**Cursor**

Activating the venv and installing scikit-learn:


scikit-learn is installed in your venv `ml.ci.uba.2026.venv`.

Install output:
- **scikit-learn** 1.8.0
- **joblib** 1.5.3
- **threadpoolctl** 3.6.0

numpy and scipy were already present. You can use the scratchpad and notebook with this venv.

---

**User**

How can I check the coefficients of the model?

---

**Cursor**

For the fitted `LogisticRegression` model in your scratchpad you can inspect coefficients like this:

- **Coefficients (betas):** `model.coef_`  
  - Shape is `(1, n_features)`. For your data that’s one row with the coefficients for X1, X2, X3.

- **Intercept:** `model.intercept_`  
  - A 1D array with one value (the intercept).

Example after fitting:

```python
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
# Or with feature names:
print(dict(zip(["X1", "X2", "X3"], model.coef_[0])))
```

So the linear index for unit \(i\) is  
`model.intercept_[0] + model.coef_[0] @ X_i`,  
and the predicted probability is the sigmoid of that (what `model.predict_proba` returns).

---

**User**

I want to create a function that takes a df and plots the true propensity score density and the predicted propensity score densities. Place it in scratchpad.py 

---

**Cursor**

Added `plot_propensity_densities(df, model, ax=None)` to `scratchpad.py`. It:

- Uses **true** propensities from `df["prop"]`
- Uses **predicted** propensities from `model.predict_proba(df[["X1","X2","X3"]])[:, 1]`
- Plots both as density histograms (overlaid, with legend)

Usage after fitting the model (e.g. in the notebook):

```python
%run scratchpad.py
plot_propensity_densities(df, model)
plt.show()
```

You can pass an `ax` to draw on a specific subplot. The function returns the axes so you can chain or adjust it.

---

**User**

I want to plot the kernel density, not histograms. Also, extend the limits of x axis to -0.1 and 1.1

---

**Cursor**

Summary of changes:

- **Kernel density:** Uses `scipy.stats.gaussian_kde` for both the true and predicted propensity scores, evaluated on 200 points over the x-range.
- **x-axis:** Set with `ax.set_xlim(-0.1, 1.1)`.

`scipy` is already available as a dependency of scikit-learn, so no new packages are required.

---

**User**

Can we install scipy in the same venv

---

**Cursor**

Scipy was already installed as a dependency when we installed scikit-learn. Verifying it's present in the venv:


**scipy** is already installed in `ml.ci.uba.2026.venv` (version 1.17.0). It was pulled in as a dependency of scikit-learn, so your `plot_propensity_densities` KDE code will run without any extra installs.

---

**User**

I want to create a 1x2 plot where in the first subplot I visualize the relationship between the true propensity scores (prop col in df) and X2, and the same for X3 in the second subplot. the propensity score should be in the y axis. Place the code in @scratchpad.py , don't create a function, just a simple block that I can run in a notebook cell

---

**Cursor**

Added a block at the bottom of `scratchpad.py` that:

- Builds a 1×2 figure with `plt.subplots(1, 2)`.
- **Left:** scatter of `X2` (x) vs `prop` (y).
- **Right:** scatter of `X3` (x) vs `prop` (y).

Scatter uses `s=1` and `alpha=0.3` so 100k points stay visible and quick to draw. Run the scratchpad in a notebook (e.g. `%run scratchpad.py`) after `df` exists, or copy that block into a cell.

---

**User**

Now I want to fit a regression tree. Can you write a two-liner using scikit learn's method. Place it in @scratchpad.py 

---

**Cursor**


Adding a two-liner that fits a regression tree (predicting Y from X) at the top of the file:



Added at the top of `scratchpad.py`:

```python
from sklearn.tree import DecisionTreeRegressor
tree_model = DecisionTreeRegressor().fit(df[["X1", "X2", "X3"]], df["Y"])
```

This fits a regression tree for **Y** on **X1, X2, X3**. Use it after `df` is defined (e.g. after running the DGP in the notebook). The fitted object is `tree_model`; predictions are `tree_model.predict(df[["X1","X2","X3"]])`.

---

**User**

Can you provide the code in @scratchpad.py  to visualize the tree?

---

**Cursor**


Added tree visualization to `scratchpad.py`:

- **Imports:** `plot_tree` from `sklearn.tree` and `matplotlib.pyplot` as `plt`.
- **Block:** builds a figure and calls `plot_tree(tree_model, feature_names=["X1", "X2", "X3"], ax=ax)` so node splits are labeled with X1/X2/X3.

Run the scratchpad after `df` and `tree_model` exist. If the tree is deep and the plot is too large, either increase the figure size or refit with a smaller `max_depth`, e.g. `DecisionTreeRegressor(max_depth=4)`.

