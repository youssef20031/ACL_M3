"""
LIME for tabular data (Ribeiro et al., 2016), as the `lime` package's LimeTabularExplainer does it
with its defaults, in numpy. The package isn't a runtime dependency: it pulls scikit-image and
matplotlib into the API image for a few dozen lines of arithmetic.

To explain one prediction of a black-box `predict_fn`:
  1. Discretize: every feature is cut at the quartiles of a background sample (LIME's
     QuartileDiscretizer); features with at most CATEGORICAL_MAX_VALUES distinct values (flags such
     as position or home/away) are categorical, each value its own bin; missing values form a bin.
  2. Perturb: `num_samples` rows are drawn, each feature independently from the background
     (LIME draws a bin by frequency, then a value inside it; a background value of that feature is
     the same draw from the empirical distribution). Row 0 is the instance itself.
  3. Represent each row by which features fall in the same bin as the instance (1) or not (0).
  4. Weight rows by an exponential kernel on their distance from the instance in that binary space,
     sqrt(exp(-d^2 / width^2)) with width 0.75 * sqrt(#features), LIME's default.
  5. Select the `num_features` features with the largest weighted-ridge coefficients ("highest
     weights"), then fit a weighted ridge regression (alpha 1, LIME's default regressor) on them.
A coefficient is the change in the black box's output, in its own units, when a feature is in the
instance's bin rather than elsewhere, near this instance. score is the weighted R^2 of that local
fit; local_prediction is the surrogate's output for the instance.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

CATEGORICAL_MAX_VALUES = 2
MISSING = -1


@dataclass
class LimeResult:
    intercept: float
    local_prediction: float
    model_prediction: float
    score: float
    num_samples: int
    kernel_width: float
    features: List[int]       # indices of the selected features, by |weight|, largest first
    weights: List[float]
    conditions: List[str]     # the instance's bin, e.g. "Minutes > 78" or "12 < Minutes <= 45"


class Discretizer:
    """Quartile bins of each feature, from a background sample (n_rows x n_features)."""

    def __init__(self, background: np.ndarray, names: Sequence[str]):
        self.names = list(names)
        self.categorical: List[Optional[np.ndarray]] = []
        self.edges: List[Optional[np.ndarray]] = []
        for j in range(background.shape[1]):
            col = background[:, j]
            values = np.unique(col[~np.isnan(col)])
            if len(values) <= CATEGORICAL_MAX_VALUES:
                self.categorical.append(values)
                self.edges.append(None)
            else:
                self.categorical.append(None)
                self.edges.append(np.unique(np.nanpercentile(col, [25, 50, 75])))

    def bins(self, x: np.ndarray) -> np.ndarray:
        """Bin index of every value (rows x features); MISSING for NaN; -2 for an unseen category."""
        x = np.atleast_2d(x)
        out = np.empty(x.shape, dtype=np.int64)
        for j in range(x.shape[1]):
            col = x[:, j]
            if self.categorical[j] is not None and not len(self.categorical[j]):
                # NaN in every background row (e.g. v4's odds before the prices are posted): every
                # value is an unseen category, and NaN is marked missing below
                out[:, j] = -2
            elif self.categorical[j] is not None:
                values = self.categorical[j]
                idx = np.searchsorted(values, col)
                hit = (idx < len(values)) & (values[np.minimum(idx, len(values) - 1)] == col)
                out[:, j] = np.where(hit, idx, -2)
            else:
                out[:, j] = np.searchsorted(self.edges[j], col, side="left")
            out[np.isnan(col), j] = MISSING
        return out

    def condition(self, j: int, b: int, label: Optional[str] = None) -> str:
        name = label or self.names[j]
        if b == MISSING:
            return f"{name} is missing"
        if self.categorical[j] is not None:
            values = self.categorical[j]
            return f"{name} = {_fmt(values[b])}" if 0 <= b < len(values) else f"{name} is unusual"
        edges = self.edges[j]
        if b == 0:
            return f"{name} <= {_fmt(edges[0])}"
        if b >= len(edges):
            return f"{name} > {_fmt(edges[-1])}"
        return f"{_fmt(edges[b - 1])} < {name} <= {_fmt(edges[b])}"


def _fmt(v: float) -> str:
    return f"{float(v):.3g}"


def weighted_ridge(z: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float):
    """Ridge regression with an unpenalised intercept and sample weights (as scikit-learn's Ridge)."""
    sw = w / w.sum()
    z_mean, y_mean = sw @ z, sw @ y
    zc, yc = z - z_mean, y - y_mean
    a = zc.T @ (zc * w[:, None]) + alpha * np.eye(z.shape[1])
    coef = np.linalg.solve(a, zc.T @ (w * yc))
    return coef, float(y_mean - z_mean @ coef)


def explain(x: np.ndarray, background: np.ndarray, predict_fn: Callable[[np.ndarray], np.ndarray],
            names: Sequence[str], labels: Optional[Sequence[str]] = None, num_samples: int = 5000,
            num_features: int = 10, kernel_width: Optional[float] = None, seed: int = 0) -> LimeResult:
    """LIME explanation of predict_fn at x (1-D), perturbing with the rows of `background`."""
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    n_feat = len(x)
    width = float(kernel_width or 0.75 * np.sqrt(n_feat))
    rng = np.random.default_rng(seed)
    disc = Discretizer(background, names)

    rows = rng.integers(0, len(background), size=(num_samples, n_feat))
    samples = background[rows, np.arange(n_feat)]
    samples[0] = x
    z = (disc.bins(samples) == disc.bins(x)).astype(float)
    distance = np.sqrt(((1.0 - z) ** 2).sum(axis=1))
    w = np.sqrt(np.exp(-(distance ** 2) / width ** 2))
    y = np.asarray(predict_fn(samples), dtype=float)

    coef, _ = weighted_ridge(z, y, w, alpha=0.01)
    k = min(num_features, n_feat)
    chosen = sorted(np.argsort(-np.abs(coef), kind="mergesort")[:k].tolist())
    coef, intercept = weighted_ridge(z[:, chosen], y, w, alpha=1.0)
    fitted = intercept + z[:, chosen] @ coef
    y_mean = np.average(y, weights=w)
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    score = 1.0 - float(np.sum(w * (y - fitted) ** 2)) / ss_tot if ss_tot > 0 else 0.0

    order = np.argsort(-np.abs(coef), kind="mergesort")
    x_bins = disc.bins(x)[0]
    labels = list(labels) if labels is not None else list(names)
    return LimeResult(
        intercept=intercept, local_prediction=float(fitted[0]), model_prediction=float(y[0]), score=score,
        num_samples=num_samples, kernel_width=width,
        features=[chosen[i] for i in order], weights=[float(coef[i]) for i in order],
        conditions=[disc.condition(chosen[i], int(x_bins[chosen[i]]), labels[chosen[i]]) for i in order],
    )
