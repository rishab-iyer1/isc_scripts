import numpy as np
import nibabel as nib
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from variance_partitioning import (
    ALPHAS as _ALPHAS_FULL,
    SUBSETS,
    UNIQUE_KEYS,
    fit_pnm_alpha,
    compute_all_subset_r2s_fixed_alpha,
    compute_unique_r2,
    compute_vp_components,
    run_permutation_vp,
    compute_significance,
    compute_per_emotion_masks,
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


def test_fit_pnm_alpha_returns_per_target_alphas_from_grid():
    """fit_pnm_alpha returns (n_targets,) array; every value must be drawn from
    the supplied alpha grid (RidgeCV selects from the grid, not interpolates)."""
    X = RNG.randn(N_SAMPLES, 3)
    Y = RNG.randn(N_SAMPLES, N_TARGETS)
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    assert best_alphas.shape == (N_TARGETS,), \
        f'expected ({N_TARGETS},), got {best_alphas.shape}'
    for a in best_alphas:
        assert np.any(np.isclose(a, ALPHAS)), \
            f'returned alpha {a} not in supplied grid {ALPHAS}'


def test_compute_all_subset_r2s_fixed_alpha_returns_dict_per_subset(p_only_data):
    """The function returns a dict keyed by every subset name in SUBSETS, each
    value (n_targets,). R²(PNM) should beat R²(N) and R²(M) when P is the only
    true predictor (sanity that the same per-target α is producing comparable
    R²s across subsets — no per-subset re-CV)."""
    X, Y = p_only_data
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    r2_dict = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=3)
    assert set(r2_dict.keys()) == set(SUBSETS.keys()), \
        f'expected keys {set(SUBSETS.keys())}, got {set(r2_dict.keys())}'
    for name, r2 in r2_dict.items():
        assert r2.shape == (N_TARGETS,), f'{name}: shape {r2.shape}'
    assert r2_dict['PNM'].mean() > r2_dict['N'].mean(), \
        f"R²(PNM) {r2_dict['PNM'].mean():.3f} should exceed R²(N) {r2_dict['N'].mean():.3f}"
    assert r2_dict['PNM'].mean() > r2_dict['M'].mean(), \
        f"R²(PNM) {r2_dict['PNM'].mean():.3f} should exceed R²(M) {r2_dict['M'].mean():.3f}"


def test_vp_components_sum_to_full_r2(p_only_data):
    X, Y = p_only_data
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    r2_dict = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=3)
    components = compute_vp_components(r2_dict)
    total = sum(components.values())
    assert np.allclose(total, r2_dict['PNM'], atol=1e-6), \
        f"sum of VP components should equal R²(PNM), max diff: {np.max(np.abs(total - r2_dict['PNM']))}"


def test_unique_zero_when_predictors_identical(identical_pn_data):
    X, Y = identical_pn_data
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    r2_dict = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=3)
    components = compute_vp_components(r2_dict)
    assert np.all(components['unique_P'] < 0.05), \
        f"unique_P should be ~0 when P==N, got mean={components['unique_P'].mean():.4f}"
    assert np.all(components['unique_N'] < 0.05), \
        f"unique_N should be ~0 when P==N, got mean={components['unique_N'].mean():.4f}"


def test_unique_nonzero_when_only_predictor(p_only_data):
    X, Y = p_only_data
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    r2_dict = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=3)
    components = compute_vp_components(r2_dict)
    assert np.mean(components['unique_P']) > 0.3, \
        f"unique_P should be large when P is only predictor, got {np.mean(components['unique_P']):.4f}"
    assert np.mean(np.abs(components['unique_N'])) < 0.1, \
        f"unique_N should be ~0 when N is noise, got {np.mean(components['unique_N']):.4f}"


def test_permutation_null_shape():
    """run_permutation_vp returns a null distribution for each of the 6 unique
    sub-model components (3 single-emotion + 3 pair). UNIQUE_KEYS pins the
    canonical key set and ordering."""
    X = RNG.randn(N_SAMPLES, 3)
    Y = RNG.randn(N_SAMPLES, N_TARGETS)
    n_perms = 5
    Y_perms = RNG.randn(n_perms, N_SAMPLES, N_TARGETS)
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    null = run_permutation_vp(X, Y_perms, best_alphas)
    assert set(null.keys()) == set(UNIQUE_KEYS), \
        f'expected {set(UNIQUE_KEYS)}, got {set(null.keys())}'
    assert UNIQUE_KEYS == ['U_P', 'U_N', 'U_M', 'U_PN', 'U_PM', 'U_NM'], \
        f'UNIQUE_KEYS ordering drifted: {UNIQUE_KEYS}'
    for key in UNIQUE_KEYS:
        assert null[key].shape == (n_perms, N_TARGETS), \
            f'{key}: expected ({n_perms}, {N_TARGETS}), got {null[key].shape}'


def test_compute_unique_r2_definitions(p_only_data):
    """compute_unique_r2 must return exactly the 6 sub-model differences:
    U_P = PNM-NM, U_N = PNM-PM, U_M = PNM-PN, U_PN = PNM-M, U_PM = PNM-N, U_NM = PNM-P."""
    X, Y = p_only_data
    best_alphas = fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=3)
    r2 = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=3)
    unique = compute_unique_r2(r2)
    assert set(unique.keys()) == set(UNIQUE_KEYS)
    np.testing.assert_allclose(unique['U_P'],  r2['PNM'] - r2['NM'])
    np.testing.assert_allclose(unique['U_N'],  r2['PNM'] - r2['PM'])
    np.testing.assert_allclose(unique['U_M'],  r2['PNM'] - r2['PN'])
    np.testing.assert_allclose(unique['U_PN'], r2['PNM'] - r2['M'])
    np.testing.assert_allclose(unique['U_PM'], r2['PNM'] - r2['N'])
    np.testing.assert_allclose(unique['U_NM'], r2['PNM'] - r2['P'])


def test_compute_per_emotion_masks_shape_and_logic():
    """Per-emotion masks: each is the strict-less-than-alpha boolean slice of
    its emotion's column in p_fdr. No union across emotions."""
    p_fdr = np.array([
        [0.01, 0.5,  0.5 ],   # sig in P only
        [0.5,  0.04, 0.5 ],   # sig in N only
        [0.5,  0.5,  0.02],   # sig in M only
        [0.5,  0.5,  0.5 ],   # sig in none
        [0.01, 0.02, 0.03],   # sig in all
        [0.05, 0.05, 0.05],   # boundary (not strictly < alpha)
    ])
    masks = compute_per_emotion_masks(p_fdr, alpha=0.05)
    assert set(masks.keys()) == {'P', 'N', 'M'}, \
        f"expected keys {{'P','N','M'}}, got {set(masks.keys())}"
    for e in ['P', 'N', 'M']:
        assert masks[e].shape == (6,), f'{e}: shape {masks[e].shape}'
        assert masks[e].dtype == bool, f'{e}: dtype {masks[e].dtype}'
    np.testing.assert_array_equal(masks['P'], np.array([True, False, False, False, True, False]))
    np.testing.assert_array_equal(masks['N'], np.array([False, True, False, False, True, False]))
    np.testing.assert_array_equal(masks['M'], np.array([False, False, True, False, True, False]))


def _write_synthetic_parc_and_4d(tmp_path, parcel_values):
    """Build a tiny (4,4,4) parc NIfTI with labels 1..n and a 4D NIfTI whose
    values are constant within each parcel, the same way isps.py paints them.

    parcel_values shape: (n_parcels, n_emotions). Returns (parc_path, vol_path, affine).
    """
    n_parcels, n_emotions = parcel_values.shape
    affine = np.eye(4)
    parc_data = np.zeros((4, 4, 4), dtype=np.int32)
    flat = parc_data.reshape(-1)
    for p in range(1, n_parcels + 1):
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
