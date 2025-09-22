import os
import sys
import numpy as np
import pickle
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import multipletests

# ensure repo root is on sys.path so `isc_scripts` package can be imported when
# pytest runs from anywhere. The repo root is two levels up from this test file.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isc_scripts import debug_utils as dbg


def make_synthetic_x(n_windows=20, n_parcels=10, n_shifts=50, random_state=0):
    rng = np.random.RandomState(random_state)
    # observed: shape (n_windows, n_parcels)
    observed = rng.randn(n_windows, n_parcels) * 0.2
    # introduce a real relationship for parcel 0 with a synthetic behavior
    # Make sure ISC values are in [-1,1]
    observed = np.clip(observed, -1, 1)
    # null distribution: shape (n_shifts, n_windows, n_parcels)
    dist = rng.randn(n_shifts, n_windows, n_parcels) * 0.2
    dist = np.clip(dist, -1, 1)
    # p-values placeholder (not used widely here)
    p = np.zeros_like(observed)
    x = {'wholebrain': [observed, p, dist]}
    return x


def test_ridge_and_permutation_consistency():
    # generate synthetic x structure similar to sliding permutation outputs
    x = make_synthetic_x(n_windows=30, n_parcels=8, n_shifts=10000, random_state=1)

    # create synthetic behavior with a relationship to parcel 0
    rng = np.random.RandomState(2)
    n_windows = x['wholebrain'][0].shape[0]
    behav = rng.randn(n_windows, 3) * 0.1
    behav[:, 0] += x['wholebrain'][0][:, 0] * 1.5  # strong relation with parcel 0

    # Fit RidgeCV on observed for each parcel separately (matching main script pattern)
    alpha_range = np.logspace(-3, 3, 20)
    true_coefs = np.zeros((x['wholebrain'][0].shape[1], behav.shape[1]))
    for parcel in range(x['wholebrain'][0].shape[1]):
        y = x['wholebrain'][0][:, parcel]
        model = RidgeCV(alphas=alpha_range, store_cv_values=True)
        model.fit(behav, y)
        true_coefs[parcel] = model.coef_

    # Now compute permuted coefs with same alpha grid (simulate what main script should do)
    n_shifts = x['wholebrain'][2].shape[0]
    perm_coefs = np.zeros((x['wholebrain'][0].shape[1], n_shifts, behav.shape[1]))
    for s in range(n_shifts):
        shifted_isc = x['wholebrain'][2][s]  # shape (n_windows, n_parcels)
        for parcel in range(shifted_isc.shape[1]):
            y = shifted_isc[:, parcel]
            model = RidgeCV(alphas=alpha_range, store_cv_values=True)
            model.fit(behav, y)
            perm_coefs[parcel, s] = model.coef_

    # sanity: perm_coefs shape matches
    assert perm_coefs.shape == (x['wholebrain'][0].shape[1], n_shifts, behav.shape[1])

    # Compute p-value for parcel 0, emotion 0
    obs = true_coefs[0, 0]
    null_dist = perm_coefs[0, :, 0]
    pval = (np.sum(np.abs(null_dist) >= abs(obs)) + 1) / (null_dist.size + 1)
    # since we implanted a signal, expect pval to be small
    assert pval < 0.05

    # Build pval array for all parcels/emotions and perform FDR
    pvals = np.empty((true_coefs.shape[0], true_coefs.shape[1]))
    for i in range(true_coefs.shape[0]):
        for j in range(true_coefs.shape[1]):
            null = perm_coefs[i, :, j]
            obs = true_coefs[i, j]
            pvals[i, j] = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)

    # Flatten and FDR
    flat = pvals.flatten()
    reject, qvals, _, _ = multipletests(flat, alpha=0.05, method='fdr_bh')
    print(flat[0:5])
    print(qvals[0:5])
    # We expect at least one significant parcel (the one we planted)
    assert reject.sum() >= 1
