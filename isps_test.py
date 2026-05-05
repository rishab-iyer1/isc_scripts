import os
import pickle
import numpy as np
import pytest
from scipy.signal import hilbert
from ISC_Helper import _compute_isps, _compute_isps_behav_corr, _compute_phaseshift_isps


def _windowed_isc(data, window_size=20, step_size=5):
    """Mean pairwise Pearson r within sliding windows. Returns (n_windows, n_voxels)."""
    n_trs, n_vox, n_subj = data.shape
    n_windows = (n_trs - window_size) // step_size + 1
    out = np.zeros((n_windows, n_vox))
    pairs = [(s1, s2) for s1 in range(n_subj) for s2 in range(s1 + 1, n_subj)]
    for i in range(n_windows):
        win = data[i * step_size:i * step_size + window_size]  # (window, n_vox, n_subj)
        for v in range(n_vox):
            rs = [np.corrcoef(win[:, v, s1], win[:, v, s2])[0, 1] for s1, s2 in pairs]
            out[i, v] = np.mean(rs)
    return out

N_TRS, N_VOXELS, N_SUBJECTS = 20, 100, 5
FREQ = 0.1  # Hz; integer frequency index for N_TRS=20 → minimal Hilbert artifacts


def _narrowband(n_trs, n_voxels, n_subjects, rng, noise_std=0.1):
    """Narrowband fMRI-like signals: sinusoid with random per-(voxel, subject) phase + small noise."""
    t = np.arange(n_trs)
    phases = rng.uniform(0, 2 * np.pi, (1, n_voxels, n_subjects))
    return np.sin(2 * np.pi * FREQ * t[:, None, None] + phases) + noise_std * rng.randn(n_trs, n_voxels, n_subjects)


@pytest.fixture
def random_data():
    return _narrowband(N_TRS, N_VOXELS, N_SUBJECTS, np.random.RandomState(0))


@pytest.fixture
def perfect_sync_data():
    """All subjects share identical narrowband signal — ISPS should be 1."""
    rng = np.random.RandomState(1)
    phases = rng.uniform(0, 2 * np.pi, (1, N_VOXELS, 1))
    signal = np.sin(2 * np.pi * FREQ * np.arange(N_TRS)[:, None, None] + phases)
    return np.tile(signal, (1, 1, N_SUBJECTS))


@pytest.fixture
def independent_data():
    """Fully independent narrowband subjects — ISPS should be near chance."""
    return _narrowband(N_TRS, N_VOXELS, N_SUBJECTS, np.random.RandomState(2))


def test_rejects_unacknowledged_input(random_data):
    with pytest.raises(ValueError, match="narrowband"):
        _compute_isps(random_data)


def test_output_shape(random_data):
    assert _compute_isps(random_data, assume_narrowband=True).shape == (N_TRS, N_VOXELS)


def test_bounded(random_data):
    result = _compute_isps(random_data, assume_narrowband=True)
    assert np.all(result >= -1e-10) and np.all(result <= 1 + 1e-10)


def test_perfect_sync(perfect_sync_data):
    assert np.allclose(_compute_isps(perfect_sync_data, assume_narrowband=True), 1.0, atol=1e-10)


def test_amplitude_invariance(random_data):
    assert np.allclose(
        _compute_isps(random_data, assume_narrowband=True),
        _compute_isps(random_data * 1000.0, assume_narrowband=True),
        atol=1e-10
    )


def test_subject_permutation_invariance(random_data):
    perm = np.random.RandomState(3).permutation(N_SUBJECTS)
    assert np.allclose(
        _compute_isps(random_data, assume_narrowband=True),
        _compute_isps(random_data[:, :, perm], assume_narrowband=True),
        atol=1e-10
    )


def test_output_is_real(random_data):
    assert np.isrealobj(_compute_isps(random_data, assume_narrowband=True))


def test_voxelwise_independence(random_data):
    """Each voxel is processed independently — single-voxel result matches the full array."""
    full = _compute_isps(random_data, assume_narrowband=True)
    single = _compute_isps(random_data[:, 0:1, :], assume_narrowband=True)
    assert np.allclose(full[:, 0], single[:, 0], atol=1e-12)


def test_global_phase_shift_invariance():
    """Adding the same phase offset to all subjects should leave PLV unchanged."""
    rng = np.random.RandomState(1234)
    t = np.arange(N_TRS)
    phases = rng.uniform(0, 2 * np.pi, (1, N_VOXELS, N_SUBJECTS))
    data    = np.sin(2 * np.pi * FREQ * t[:, None, None] + phases)
    shifted = np.sin(2 * np.pi * FREQ * t[:, None, None] + phases + 0.7)
    np.testing.assert_allclose(
        _compute_isps(data,    assume_narrowband=True),
        _compute_isps(shifted, assume_narrowband=True),
        atol=1e-8
    )


def test_antiphase_two_subjects():
    """Exactly antiphase signals (N=2) should give PLV ≈ 0 away from edges."""
    t = np.arange(N_TRS)
    sig = np.sin(2 * np.pi * FREQ * t)[:, None, None]  # (T, 1, 1)
    data = np.concatenate([sig, -sig], axis=2)           # subj 2 is antiphase
    result = _compute_isps(data, assume_narrowband=True)
    assert np.mean(result[N_TRS // 4 : 3 * N_TRS // 4]) < 0.05


def test_matches_reference_formula(random_data):
    """Output must match the Glerean PLV formula exactly."""
    phases = np.angle(hilbert(random_data, axis=0))
    reference = np.abs(np.mean(np.exp(1j * phases), axis=2))
    np.testing.assert_allclose(_compute_isps(random_data, assume_narrowband=True), reference, atol=1e-10)


# --- _compute_isps_behav_corr ---

N_TRS_BEHAV, N_VOXELS_BEHAV, N_EMOTIONS = 50, 10, 3


def test_isps_behav_corr_shape():
    rng = np.random.RandomState(10)
    isps = rng.randn(N_TRS_BEHAV, N_VOXELS_BEHAV)
    behav = rng.randn(N_TRS_BEHAV, N_EMOTIONS)
    assert _compute_isps_behav_corr(isps, behav).shape == (N_VOXELS_BEHAV, N_EMOTIONS)


def test_isps_behav_corr_perfect():
    """Identical signals → r = 1.0."""
    signal = np.random.RandomState(11).randn(N_TRS_BEHAV)
    r = _compute_isps_behav_corr(signal[:, None], signal[:, None])
    assert np.isclose(r[0, 0], 1.0)


def test_isps_behav_corr_anticorr():
    """Antiphase signals → r = -1.0."""
    signal = np.random.RandomState(12).randn(N_TRS_BEHAV)
    r = _compute_isps_behav_corr(signal[:, None], -signal[:, None])
    assert np.isclose(r[0, 0], -1.0)


def test_isps_behav_corr_bounded():
    rng = np.random.RandomState(13)
    r = _compute_isps_behav_corr(rng.randn(N_TRS_BEHAV, N_VOXELS_BEHAV),
                                  rng.randn(N_TRS_BEHAV, N_EMOTIONS))
    assert np.all(r >= -1 - 1e-10) and np.all(r <= 1 + 1e-10)


def test_isps_behav_corr_amplitude_invariance():
    rng = np.random.RandomState(14)
    isps = rng.randn(N_TRS_BEHAV, N_VOXELS_BEHAV)
    behav = rng.randn(N_TRS_BEHAV, N_EMOTIONS)
    np.testing.assert_allclose(
        _compute_isps_behav_corr(isps, behav),
        _compute_isps_behav_corr(isps * 100, behav * 0.01),
        atol=1e-10
    )


# --- Convergence: ISPS and windowed ISC should measure the same construct ---

def test_isps_isc_convergence():
    """ISPS and windowed ISC should correlate strongly across voxels with varying synchrony.

    Each voxel has a different synchrony level: data = alpha * shared + (1-alpha) * noise,
    where alpha is swept from 0 (fully independent) to 1 (perfectly synchronized).
    Both measures are time-averaged per voxel; the resulting (n_voxels,) vectors are
    compared via Spearman r — should be > 0.95 for narrowband signals.
    """
    from scipy.stats import spearmanr

    n_trs, n_subj, n_vox = 454, 5, 20
    freq = 0.055  # Hz — center of Glerean 0.04–0.07 Hz band
    t = np.arange(n_trs)
    rng = np.random.RandomState(99)
    alphas = np.linspace(0, 1, n_vox)  # synchrony level per voxel

    shared = np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))  # (n_trs,)
    noise  = rng.randn(n_trs, n_vox, n_subj)
    shared_broadcast = np.broadcast_to(shared[:, None, None], (n_trs, n_vox, n_subj))
    data = alphas[None, :, None] * shared_broadcast + (1 - alphas[None, :, None]) * noise

    mean_isps = _compute_isps(data, assume_narrowband=True).mean(axis=0)   # (n_vox,)
    mean_isc  = _windowed_isc(data, window_size=30, step_size=1).mean(axis=0)  # (n_vox,)

    r, _ = spearmanr(mean_isps, mean_isc)
    assert r > 0.95, f"ISPS–ISC Spearman r = {r:.3f}, expected > 0.95"


# --- Empirical convergence: ISPS vs. temporal ISC on real onesmallstep data ---

_DATA_ROOT = '/usr/people/ri4541/juke/isc/outputs/onesmallstep/data'
_SCRIPTS_ROOT = '/mnt/cup/labs/norman/rsiyer/isc/isc_scripts'
_REAL_DATA_FILES = [
    f'{_DATA_ROOT}/isps_27subj_1000parcels_1rois.pkl',
    f'{_DATA_ROOT}/median_isc.npy',
    f'{_DATA_ROOT}/mask_img.npy',
    f'{_SCRIPTS_ROOT}/schaefer_2018/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz',
]


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in _REAL_DATA_FILES),
    reason="real onesmallstep data not available",
)
def test_isps_isc_empirical_convergence():
    """ISPS and temporal ISC should correlate across 1000 parcels on real data.

    Threshold is modest (0.3) because temporal ISC uses the full preprocessed
    spectrum while ISPS is narrowband (0.04–0.07 Hz); the two measures are
    related but not equivalent in frequency content.
    """
    from scipy.stats import spearmanr
    from ISC_Helper import load_schaeffer1000

    with open(_REAL_DATA_FILES[0], 'rb') as f:
        isps_roi = pickle.load(f)
    mean_isps = isps_roi['wholebrain'].mean(axis=0)  # (1000,)

    median_isc_vox = np.load(_REAL_DATA_FILES[1])    # (n_voxels,)
    _, masked_parc = load_schaeffer1000(_REAL_DATA_FILES[3], _REAL_DATA_FILES[2])
    isc_parcellated = np.array([
        median_isc_vox[masked_parc == p].mean() for p in range(1, 1001)
    ])

    r, _ = spearmanr(mean_isps, isc_parcellated)
    assert r > 0.3, f"ISPS–ISC Spearman r = {r:.3f}, expected > 0.3"


# --- _compute_phaseshift_isps ---

_N_TRS_PH, _N_VOX_PH, _N_SUBJ_PH, _N_SHIFTS = 40, 5, 4, 10


@pytest.fixture
def narrowband_perm():
    return _narrowband(_N_TRS_PH, _N_VOX_PH, _N_SUBJ_PH, np.random.RandomState(50))


def test_phaseshift_isps_shapes(narrowband_perm):
    observed, distribution = _compute_phaseshift_isps(narrowband_perm, n_shifts=_N_SHIFTS, random_state=42)
    assert observed.shape == (_N_TRS_PH, _N_VOX_PH)
    assert distribution.shape == (_N_SHIFTS, _N_TRS_PH, _N_VOX_PH)


def test_phaseshift_isps_observed_matches_direct(narrowband_perm):
    observed, _ = _compute_phaseshift_isps(narrowband_perm, n_shifts=1, random_state=42)
    direct = _compute_isps(narrowband_perm, assume_narrowband=True)
    np.testing.assert_allclose(observed, direct, atol=1e-12)


def test_phaseshift_isps_distribution_bounded(narrowband_perm):
    _, distribution = _compute_phaseshift_isps(narrowband_perm, n_shifts=_N_SHIFTS, random_state=42)
    assert np.all(distribution >= -1e-10) and np.all(distribution <= 1 + 1e-10)


def test_phaseshift_isps_distribution_varies(narrowband_perm):
    """Different permutations produce different ISPS values."""
    _, distribution = _compute_phaseshift_isps(narrowband_perm, n_shifts=_N_SHIFTS, random_state=42)
    assert distribution.std() > 1e-6


def test_phaseshift_isps_perfect_sync_null(perfect_sync_data):
    """Observed ISPS on synchronized data should exceed the phase-randomized null."""
    observed, distribution = _compute_phaseshift_isps(perfect_sync_data, n_shifts=20, random_state=42)
    assert observed.mean() > distribution.mean() + 0.05
