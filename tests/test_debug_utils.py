import numpy as np
import os
import sys

# ensure repo root is on sys.path so `isc_scripts` package can be imported when
# pytest runs from anywhere. The repo root is two levels up from this test file.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isc_scripts import debug_utils as dbg


def test_ridge_recovery_basic():
    res = dbg.ridge_recovery_test(n_samples=300, n_features=15, n_informative=4, random_state=42)
    # basic sanity checks
    assert 'true_coefs' in res and 'est_coefs' in res
    assert res['est_coefs'].shape[0] == res['true_coefs'].shape[0]
    # R2 should be positive for signal present
    assert res['r2'] > 0.1


def test_permutation_strategy_sanity():
    rng = np.random.RandomState(0)
    n = 200
    p = 50
    e = 3
    # Simulate isc: random features
    isc = rng.randn(n, p)
    # Simulate behavior weakly related to first 5 features
    true_coefs = np.zeros(p)
    true_coefs[:5] = [1, -0.5, 0.7, 0.2, -0.3]
    behav = isc.dot(true_coefs)[:, None] + rng.randn(n, 1) * 0.5
    behav = np.hstack([behav for _ in range(e)])

    observed, nulls = dbg.validate_permutation_strategy(isc, behav, n_perm=100, random_state=1)
    # shapes
    assert observed.shape == (p, e)
    assert nulls.shape[1:] == observed.shape
    # null mean should be near zero
    assert np.max(np.abs(nulls.mean(axis=0))) < 0.2
