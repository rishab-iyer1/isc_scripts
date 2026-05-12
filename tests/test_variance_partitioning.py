import numpy as np
import nibabel as nib
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from variance_partitioning import (
    fit_subset_r2,
    compute_all_subset_r2s,
    compute_vp_components,
    fit_subset_r2_fixed_alpha,
    run_permutation_vp,
    compute_significance,
    compute_beta_sig_mask,
    load_isps_pvals_per_parcel,
    load_isps_betas_per_parcel,
)

BASE_DIR = '/usr/people/ri4541/juke/isc'
PVALS_PATH = f'{BASE_DIR}/outputs/onesmallstep/data/isps_emotion_pvals_corrected.nii.gz'
BETAS_PATH = f'{BASE_DIR}/outputs/onesmallstep/data/isps_emotion_betas_corrected.nii.gz'
PARC_PATH = f'{BASE_DIR}/isc_scripts/schaefer_2018/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'

RNG = np.random.RandomState(42)
N_SAMPLES, N_TARGETS = 100, 20
ALPHAS = np.logspace(-3, 3, 10)


@pytest.fixture
def orthogonal_data():
    """Three orthogonal predictors, each independently predicts a different target group."""
    rng = np.random.RandomState(0)
    X = np.zeros((N_SAMPLES, 3))
    X[:, 0] = rng.randn(N_SAMPLES)
    X[:, 1] = rng.randn(N_SAMPLES)
    X[:, 2] = rng.randn(N_SAMPLES)
    # Orthogonalize
    X[:, 1] -= X[:, 0] * (X[:, 0] @ X[:, 1]) / (X[:, 0] @ X[:, 0])
    X[:, 2] -= X[:, 0] * (X[:, 0] @ X[:, 2]) / (X[:, 0] @ X[:, 0])
    X[:, 2] -= X[:, 1] * (X[:, 1] @ X[:, 2]) / (X[:, 1] @ X[:, 1])
    Y = X[:, 0:1] * 3 + rng.randn(N_SAMPLES, N_TARGETS) * 0.1
    return X, Y


@pytest.fixture
def p_only_data():
    """Only P predicts Y; N and M are pure noise."""
    rng = np.random.RandomState(1)
    X = rng.randn(N_SAMPLES, 3)
    Y = X[:, 0:1] * 5 + rng.randn(N_SAMPLES, N_TARGETS) * 0.1
    return X, Y


@pytest.fixture
def identical_pn_data():
    """P and N are identical predictors → unique_P ≈ unique_N ≈ 0."""
    rng = np.random.RandomState(2)
    shared = rng.randn(N_SAMPLES)
    X = np.column_stack([shared, shared, rng.randn(N_SAMPLES)])
    Y = shared[:, None] * 3 + rng.randn(N_SAMPLES, N_TARGETS) * 0.1
    return X, Y


def test_fit_subset_r2_output_shape():
    X = RNG.randn(N_SAMPLES, 3)
    Y = RNG.randn(N_SAMPLES, N_TARGETS)
    r2, best_alphas = fit_subset_r2(X, Y, [0, 1, 2], ALPHAS, cv=3)
    assert r2.shape == (N_TARGETS,), f"expected ({N_TARGETS},), got {r2.shape}"
    assert best_alphas.shape == (N_TARGETS,)


def test_vp_components_sum_to_full_r2(p_only_data):
    X, Y = p_only_data
    r2_dict, _ = compute_all_subset_r2s(X, Y, ALPHAS, cv=3)
    components = compute_vp_components(r2_dict)
    total = sum(components.values())
    assert np.allclose(total, r2_dict['PNM'], atol=1e-6), \
        f"sum of VP components should equal R²(PNM), max diff: {np.max(np.abs(total - r2_dict['PNM']))}"


def test_unique_zero_when_predictors_identical(identical_pn_data):
    X, Y = identical_pn_data
    r2_dict, _ = compute_all_subset_r2s(X, Y, ALPHAS, cv=3)
    components = compute_vp_components(r2_dict)
    # When P == N, neither has unique variance above the other
    assert np.all(components['unique_P'] < 0.05), \
        f"unique_P should be ~0 when P==N, got mean={components['unique_P'].mean():.4f}"
    assert np.all(components['unique_N'] < 0.05), \
        f"unique_N should be ~0 when P==N, got mean={components['unique_N'].mean():.4f}"


def test_unique_nonzero_when_only_predictor(p_only_data):
    X, Y = p_only_data
    r2_dict, _ = compute_all_subset_r2s(X, Y, ALPHAS, cv=3)
    components = compute_vp_components(r2_dict)
    # P is the only true predictor → unique_P should be large
    assert np.mean(components['unique_P']) > 0.3, \
        f"unique_P should be large when P is only predictor, got {np.mean(components['unique_P']):.4f}"
    # N and M are noise → their unique components should be near zero
    assert np.mean(np.abs(components['unique_N'])) < 0.1, \
        f"unique_N should be ~0 when N is noise, got {np.mean(components['unique_N']):.4f}"


def test_permutation_null_shape():
    X = RNG.randn(N_SAMPLES, 3)
    Y = RNG.randn(N_SAMPLES, N_TARGETS)
    n_perms = 5
    Y_perms = RNG.randn(n_perms, N_SAMPLES, N_TARGETS)
    _, alpha_dict = compute_all_subset_r2s(X, Y, ALPHAS, cv=3)
    null = run_permutation_vp(X, Y_perms, alpha_dict)
    for key in ['unique_P', 'unique_N', 'unique_M']:
        assert key in null, f"missing key {key}"
        assert null[key].shape == (n_perms, N_TARGETS), \
            f"{key}: expected ({n_perms}, {N_TARGETS}), got {null[key].shape}"


def test_compute_beta_sig_mask_shape_and_logic():
    """Union semantics: sig in any emotion → mask True; sig in none → False."""
    p_fdr = np.array([
        [0.01, 0.5,  0.5 ],   # sig in P only       → True
        [0.5,  0.04, 0.5 ],   # sig in N only       → True
        [0.5,  0.5,  0.02],   # sig in M only       → True
        [0.5,  0.5,  0.5 ],   # sig in none         → False
        [0.01, 0.02, 0.03],   # sig in all          → True
        [0.05, 0.05, 0.05],   # boundary (not < α)  → False
    ])
    mask = compute_beta_sig_mask(p_fdr, alpha=0.05)
    expected = np.array([True, True, True, False, True, False])
    assert mask.shape == (6,), f"expected (6,), got {mask.shape}"
    assert mask.dtype == bool, f"expected bool, got {mask.dtype}"
    assert np.array_equal(mask, expected), f"got {mask}, expected {expected}"


def _write_synthetic_parc_and_4d(tmp_path, parcel_values):
    """Build a tiny (4,4,4) parc NIfTI with labels 1..n and a 4D NIfTI whose
    values are constant within each parcel, the same way isps.py paints them.

    parcel_values shape: (n_parcels, n_emotions). Returns (parc_path, vol_path,
    affine).
    """
    n_parcels, n_emotions = parcel_values.shape
    affine = np.eye(4)
    parc_data = np.zeros((4, 4, 4), dtype=np.int32)
    flat = parc_data.reshape(-1)
    for p in range(1, n_parcels + 1):
        # assign 2 voxels per parcel
        flat[2 * (p - 1):2 * p] = p
    parc_img = nib.Nifti1Image(parc_data, affine)
    parc_path = tmp_path / 'parc.nii.gz'
    nib.save(parc_img, str(parc_path))

    vol = np.zeros((4, 4, 4, n_emotions))
    for p in range(1, n_parcels + 1):
        mask = parc_data == p
        vol[mask, :] = parcel_values[p - 1]
    vol_img = nib.Nifti1Image(vol, affine)
    vol_path = tmp_path / 'vol.nii.gz'
    nib.save(vol_img, str(vol_path))
    return str(parc_path), str(vol_path), affine


def test_load_isps_pvals_per_parcel_roundtrip(tmp_path):
    n_parcels, n_emotions = 5, 3
    rng = np.random.RandomState(99)
    pvals = rng.rand(n_parcels, n_emotions)
    parc_path, vol_path, _ = _write_synthetic_parc_and_4d(tmp_path, pvals)
    out = load_isps_pvals_per_parcel(vol_path, parc_path, n_parcels=n_parcels)
    assert out.shape == (n_parcels, n_emotions)
    np.testing.assert_allclose(out, pvals)


def test_load_isps_betas_per_parcel_roundtrip(tmp_path):
    n_parcels, n_emotions = 5, 3
    rng = np.random.RandomState(101)
    betas = rng.randn(n_parcels, n_emotions) * 0.1
    parc_path, vol_path, _ = _write_synthetic_parc_and_4d(tmp_path, betas)
    out = load_isps_betas_per_parcel(vol_path, parc_path, n_parcels=n_parcels)
    assert out.shape == (n_parcels, n_emotions)
    np.testing.assert_allclose(out, betas)


@pytest.mark.skipif(
    not (os.path.exists(PVALS_PATH) and os.path.exists(BETAS_PATH)
         and os.path.exists(PARC_PATH)),
    reason='requires existing isps.py outputs and Schaefer parcellation')
def test_pvals_and_betas_corrected_agree():
    """A parcel is FDR-significant iff its thresholded beta is nonzero.

    isps.py writes p_fdr to one NIfTI and `where(p_fdr<0.05, true_coefs, 0)` to
    another. The two must agree element-wise: significant p-values must
    coincide with nonzero betas.
    """
    p_fdr = load_isps_pvals_per_parcel(PVALS_PATH, PARC_PATH, n_parcels=1000)
    betas = load_isps_betas_per_parcel(BETAS_PATH, PARC_PATH, n_parcels=1000)
    mask_from_pvals = p_fdr < 0.05
    mask_from_betas = betas != 0.0
    diff = np.argwhere(mask_from_pvals != mask_from_betas)
    assert diff.size == 0, (
        f'p-value/beta mask mismatch at {diff.shape[0]} (parcel, emotion) cells; '
        f'first few: {diff[:5].tolist()}')
