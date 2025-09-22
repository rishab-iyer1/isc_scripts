"""Debug utilities for ISC-Emotion ridge regression analysis.

Provides modular functions to validate data, test ridge regression behavior with
synthetic data, validate permutation testing, inspect p-values and corrections,
and produce diagnostic visualizations. Designed to be run step-by-step from an
interactive session or tests. Includes assertions and informative error
messages.

Usage: import isc_scripts.debug_utils as dbg
      dbg.validate_isc(isc_array)
      dbg.test_ridge_recovery()
"""
from typing import Tuple, Optional, Sequence
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
import warnings


def validate_isc(isc: np.ndarray, name: str = 'isc') -> None:
    """Validate ISC array values and shape.

    Args:
        isc: array-like of shape (n_windows, n_voxels) or (n_voxels,) or (n_windows,)
        name: optional name used in error messages

    Raises AssertionError with clear message on failure.
    """
    isc = np.asarray(isc)
    if isc.size == 0:
        raise AssertionError(f"{name} is empty")

    if not np.isfinite(isc).all():
        n_inf = np.isinf(isc).sum()
        n_nan = np.isnan(isc).sum()
        raise AssertionError(f"{name} contains non-finite values: {n_inf} inf, {n_nan} NaN")

    # Check range
    minv, maxv = np.nanmin(isc), np.nanmax(isc)
    if minv < -1.0001 or maxv > 1.0001:
        raise AssertionError(f"{name} values outside expected ISC range [-1,1]: min={minv}, max={maxv}")

    print(f"{name} ok: shape={isc.shape}, min={minv:.3f}, max={maxv:.3f}")


def validate_emotions(emotions: np.ndarray, n_trs: Optional[int] = None, name: str = 'emotions') -> None:
    """Validate emotion consensus matrix.

    Args:
        emotions: array-like shape (n_subjects?, n_trs, n_emotions) OR (n_trs, n_emotions)
        n_trs: expected number of TRs (optional)
    """
    emotions = np.asarray(emotions)
    if emotions.ndim == 3:
        # collapse subject dimension
        data = emotions.mean(axis=0)
    else:
        data = emotions

    if data.size == 0:
        raise AssertionError(f"{name} is empty")

    if n_trs is not None and data.shape[0] != n_trs:
        raise AssertionError(f"{name} has {data.shape[0]} timepoints but expected {n_trs}")

    if not np.isfinite(data).all():
        raise AssertionError(f"{name} contains non-finite values")

    # Check for very low variance across time (constant signal)
    var_per_emotion = np.nanvar(data, axis=0)
    if np.any(var_per_emotion == 0):
        idx = np.where(var_per_emotion == 0)[0]
        raise AssertionError(f"{name} has zero variance for emotion indices: {idx}")

    print(f"{name} ok: shape={data.shape}, var_min={var_per_emotion.min():.4e}, var_max={var_per_emotion.max():.4e}")


def check_predictor_variance(X: np.ndarray, tol: float = 1e-8) -> None:
    """Ensure predictors have non-zero variance and reasonable scale.

    Args:
        X: shape (n_samples, n_features)
        tol: variance tolerance below which feature is considered constant
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]

    variances = np.nanvar(X, axis=0)
    zero_var = np.where(variances <= tol)[0]
    if zero_var.size:
        raise AssertionError(f"Found constant predictors at indices: {zero_var}. Remove or regularize them.")

    print(f"All predictors have variance > {tol}. min_var={variances.min():.4e}")


def standardize_features(X: np.ndarray, return_stats: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Z-score standardize features (columns)."""
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=0)
    # avoid divide by zero
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    Xs = (X - mu) / sigma_safe
    if return_stats:
        return Xs, (mu, sigma_safe)
    return Xs, None


def ridge_recovery_test(n_samples: int = 200, n_features: int = 10, n_informative: int = 3,
                        alpha_grid: Optional[Sequence[float]] = None, random_state: Optional[int] = 0):
    """Unit test: generate synthetic data with known coefficients and test RidgeCV recovery.

    Returns dict with true_coefs, estimated_coefs, r2, and chosen alphas.
    """
    rng = np.random.RandomState(random_state)
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 50)

    # Create design matrix
    X = rng.randn(n_samples, n_features)
    true_coefs = np.zeros(n_features)
    informative_idx = np.arange(n_informative)
    true_coefs[informative_idx] = rng.randn(n_informative) * 2.0

    y = X.dot(true_coefs) + rng.randn(n_samples) * 0.5

    # Standardize features before ridge
    Xs, stats = standardize_features(X, return_stats=True)

    model = RidgeCV(alphas=alpha_grid, store_cv_values=True)
    model.fit(Xs, y)
    est = model.coef_
    r2 = model.score(Xs, y)

    # Check that informative coefficients are recovered with correct sign at least
    signs_ok = np.sign(est[informative_idx]) == np.sign(true_coefs[informative_idx])
    if not np.any(signs_ok):
        warnings.warn("Ridge did not recover informative coefficient signs; consider changing alpha grid or regularization.")

    print(f"Ridge recovery: chosen alpha={model.alpha_}, R2={r2:.3f}")
    return {"true_coefs": true_coefs, "est_coefs": est, "r2": r2, "alpha": model.alpha_}


def ridge_alpha_sweep(X: np.ndarray, y: np.ndarray, alphas: Optional[Sequence[float]] = None, cv: int = 5):
    """Fit RidgeCV over alpha grid and return coefficients for each alpha.

    Helps understand shrinkage behavior.
    """
    if alphas is None:
        alphas = np.logspace(-6, 6, 50)
    Xs, _ = standardize_features(X, return_stats=True)
    coefs = []
    scores = []
    for a in alphas:
        clf = Ridge(alpha=a)
        # simple CV with KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        scr = []
        est_coefs = []
        for tr, te in kf.split(Xs):
            clf.fit(Xs[tr], y[tr])
            scr.append(clf.score(Xs[te], y[te]))
            est_coefs.append(clf.coef_)
        coefs.append(np.mean(est_coefs, axis=0))
        scores.append(np.mean(scr))
    coefs = np.vstack(coefs)
    scores = np.array(scores)
    return np.array(alphas), coefs, scores


def validate_permutation_strategy(isc: np.ndarray, behav: np.ndarray, n_perm: int = 100, random_state: Optional[int] = 0):
    """Validate permutation by shuffling behavior labels only and building null distribution.

    Args:
        isc: shape (n_samples, n_features) where samples typically are windows
        behav: shape (n_samples, n_emotions)
    Returns:
        observed_coefs: shape (n_features, n_emotions)
        null_coefs: shape (n_perm, n_features, n_emotions)
    """
    isc = np.asarray(isc)
    behav = np.asarray(behav)
    n = isc.shape[0]
    assert behav.shape[0] == n, "ISC and behavior must have same number of samples (time windows)"

    # fit on original data
    Xs, stats = standardize_features(isc, return_stats=True)
    observed = []
    for e in range(behav.shape[1]):
        model = RidgeCV(alphas=np.logspace(-6, 6, 30))
        model.fit(Xs, behav[:, e])
        observed.append(model.coef_)
    observed = np.vstack(observed).T  # shape (n_features, n_emotions)

    # build nulls
    rng = np.random.RandomState(random_state)
    nulls = np.zeros((n_perm, observed.shape[0], observed.shape[1]))
    for i in range(n_perm):
        perm_idx = rng.permutation(n)
        for e in range(behav.shape[1]):
            model = RidgeCV(alphas=np.logspace(-6, 6, 30))
            model.fit(Xs, behav[perm_idx, e])
            nulls[i, :, e] = model.coef_

    # Sanity checks
    null_mean = nulls.mean(axis=0)
    if np.any(np.abs(null_mean) > 1e-1):
        warnings.warn(f"Null distribution mean per-coef > 0.1: max abs mean={np.max(np.abs(null_mean)):.3f}")

    print(f"Permutation completed: observed coefs shape {observed.shape}, nulls shape {nulls.shape}")
    return observed, nulls


def compare_null_real(observed: np.ndarray, nulls: np.ndarray, feature_idx: int = 0, emotion_idx: int = 0):
    """Compare distributions for one coefficient: plot and compute p-value.

    Returns p-value (two-sided) based on nulls.
    """
    obs = observed[feature_idx, emotion_idx]
    null_dist = nulls[:, feature_idx, emotion_idx]
    p = (np.sum(np.abs(null_dist) >= abs(obs)) + 1) / (null_dist.size + 1)
    print(f"Feature {feature_idx}, emotion {emotion_idx}: obs={obs:.4f}, null_mean={null_dist.mean():.4f}, p={p:.4f}")
    plt.figure()
    plt.hist(null_dist, bins=50, alpha=0.7, label='null')
    plt.axvline(obs, color='r', label='obs')
    plt.legend()
    plt.title(f"Feature {feature_idx} emotion {emotion_idx} p={p:.4f}")
    return p


def extract_raw_pvals(observed: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    """Compute two-sided p-values per coefficient from null distribution.

    observed shape: (n_features, n_emotions)
    nulls shape: (n_perm, n_features, n_emotions)
    returns pvals shape (n_features, n_emotions)
    """
    n_perm = nulls.shape[0]
    pvals = np.zeros_like(observed)
    for i in range(observed.shape[0]):
        for j in range(observed.shape[1]):
            null = nulls[:, i, j]
            obs = observed[i, j]
            p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
            pvals[i, j] = p
    return pvals


def compare_corrections(pvals: np.ndarray, alpha: float = 0.05):
    """Compare multiple comparison corrections and print counts of significant tests.

    pvals: shape (n_tests,) or (n_features, n_emotions)
    Returns dict of method->(reject, qvals)
    """
    p = np.asarray(pvals).flatten()
    methods = ['bonferroni', 'fdr_bh', 'holm']
    out = {}
    for m in methods:
        reject, q, _, _ = multipletests(p, alpha=alpha, method=m)
        out[m] = {'reject': reject.reshape(pvals.shape), 'qvals': q.reshape(pvals.shape), 'n_significant': int(reject.sum())}
        print(f"Method {m}: {out[m]['n_significant']} significant (alpha={alpha})")
    return out


def plot_pval_histogram(pvals: np.ndarray, bins: int = 50):
    p = np.asarray(pvals).flatten()
    plt.figure()
    plt.hist(p, bins=bins, range=(0, 1), color='C0', alpha=0.8)
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.title('P-value distribution')


def plot_coef_vs_isc(isc: np.ndarray, behav: np.ndarray, feature_idx: int = 0, emotion_idx: int = 0):
    """Scatter plot of ISC feature vs behavior for a single emotion/time series correlation.

    Assumes isc shape (n_samples, n_features) and behav (n_samples, n_emotions)
    """
    isc = np.asarray(isc)
    behav = np.asarray(behav)
    x = isc[:, feature_idx]
    y = behav[:, emotion_idx]
    plt.figure()
    plt.scatter(x, y, alpha=0.6)
    m, b, r, p, se = stats.linregress(x, y)
    xs = np.linspace(np.min(x), np.max(x), 50)
    plt.plot(xs, m * xs + b, color='r')
    plt.title(f'Feature {feature_idx} vs emotion {emotion_idx} r={r:.3f} p={p:.3g}')
    return r, p


def common_pitfalls() -> str:
    s = """
    Common pitfalls and checks:
    - ISC values outside [-1,1]: may indicate data scaling/normalization errors.
    - Misaligned time series: ensure behavior and ISC windows match in length and alignment.
    - Constant predictors across windows: e.g., ROI averaged to constant.
    - Standardization mismatch between real and permuted pipelines.
    - Using different alpha/grids for observed vs null leads to biased p-values.
    - Small number of permutations limits p-value resolution (use at least 1k-10k if possible).
    - Overly strict FDR scope: applying correction per-parcel vs across all tests changes results.
    Suggestion: run `validate_isc`, `validate_emotions`, then `validate_permutation_strategy` on a small subset.
    """
    return s
