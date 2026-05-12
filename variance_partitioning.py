"""
Variance partitioning of ISPS by emotion consensus (positive, negative, mixed).

Fits all 7 subset RidgeCV models and applies inclusion-exclusion to decompose
R²(PNM) into unique and shared components per parcel. Permutation testing
uses phase-shifted ISPS null distributions.

Analysis decisions
------------------
- Alpha range   : np.logspace(-3, 3, 20)  — 20 log-spaced values 0.001–1000;
                  same range as ridge.py; covers typical neuroimaging data scale.
- R² metric     : Cross-validated R² (5-fold) with per-target optimal alpha.
                  Step 1: RidgeCV selects best_alphas_ per target on full data.
                  Step 2: K-fold CV with Ridge(best_alphas_) gives unbiased R².
                  CV R² used for BOTH observed and permuted data so the null is
                  properly centered. In-sample R² inflates the null (tested and
                  rejected: null mean ~0.01 vs observed mean ~0.017, null std ~0.044
                  — no parcels survive FDR correction with in-sample R²).
- CV folds      : 5 for both alpha selection and R² estimation (n≈90/fold from 454 TRs).
- Permutation α : Per-target best_alphas_ from each observed subset fit, fixed for
                  permutations (no re-selecting alpha per permutation). Justified:
                  the alpha controls regularization, not direction of the statistic;
                  phase-shuffled null CV R² is properly centered near 0 with fixed
                  alpha. Avoids running RidgeCV 4096× for permutations.
- fit_intercept : True — emotion consensus variance is always non-negative.
- Components    : 7 Venn regions (3 unique + 3 pairwise-shared + 1 triple-shared).
                  Only unique components are permutation-tested; shared are
                  descriptive (can be negative via suppression).
- N permutations: 1024 (matches existing ISPS phase-shift null).
- FDR threshold : 0.05 (BH), consistent with ridge.py.
- Sig mask      : Union of parcels FDR-significant for ANY emotion in
                  isps_emotion_pvals_corrected.nii.gz (produced by isps.py,
                  which tests ridge coefficients — properly centered at 0 under
                  the phase-shift null). Unique components are FDR-corrected
                  only within this mask, reducing the multiple comparison
                  burden from 1000 to mask.sum() and increasing power. Parcels
                  outside the mask receive p_fdr=1.0 in output. A consistency
                  assertion confirms that the FDR-significant set in
                  isps_emotion_pvals_corrected matches the nonzero set in
                  isps_emotion_betas_corrected (must agree by isps.py
                  construction).
"""

import os
import pickle
import sys

import nibabel as nib
import numpy as np
from himalaya.ridge import Ridge, RidgeCV
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from isc_standalone import p_from_null

BASE_DIR = '/usr/people/ri4541/juke/isc'
ALPHAS = np.logspace(-3, 3, 20)
CV_FOLDS = 5

# Predictor column indices: P=0, N=1, M=2
SUBSETS = {
    'P':   [0],
    'N':   [1],
    'M':   [2],
    'PN':  [0, 1],
    'PM':  [0, 2],
    'NM':  [1, 2],
    'PNM': [0, 1, 2],
}

# The 4 subsets needed to compute the 3 unique components
PERM_SUBSETS = ['PNM', 'NM', 'PM', 'PN']


def load_vp_data(task='onesmallstep', base_dir=BASE_DIR):
    """Load X (454,3), Y (454,1000), Y_perms (1024,454,1000) for VP analysis."""
    data_dir = f'{base_dir}/outputs/{task}/data'
    label_dir = f'{base_dir}/VideoLabelling'
    isps_path = f'{data_dir}/isps_27subj_1000parcels_1rois.pkl'
    perm_path = (f'{data_dir}/sliding_isc/permutations/'
                 f'phaseshift_size30_step5_1000parcelsparcellated_1024perms_1rois_isps_x')

    coded_states = np.load(f'{label_dir}/coded_states_{task}_smooth.npy')
    coded_states = coded_states[:, :-30, :]        # trim to (27, 454, 5)
    X = np.var(coded_states[:, :, :3], axis=0)     # (454, 3): inter-subject consensus variance

    with open(isps_path, 'rb') as f:
        isps_data = pickle.load(f)
    Y = isps_data['wholebrain']                    # (454, 1000) observed ISPS

    with open(perm_path, 'rb') as f:
        perm_data = pickle.load(f)
    Y_perms = perm_data['wholebrain'][2]           # (1024, 454, 1000)

    print(f'X: {X.shape}, Y: {Y.shape}, Y_perms: {Y_perms.shape}')
    return X, Y, Y_perms


def _cv_r2_fixed_alpha(X, Y, cols, best_alphas, n_splits=CV_FOLDS):
    """K-fold CV R² using Ridge with fixed per-target alpha. Returns (n_targets,)."""
    kf = KFold(n_splits=n_splits, shuffle=False)
    r2_folds = np.zeros((n_splits, Y.shape[1]))
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx][:, cols], X[test_idx][:, cols]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        model = Ridge(alpha=best_alphas, fit_intercept=True)
        model.fit(X_tr, Y_tr)
        r2_folds[fold_idx] = np.array(model.score(X_te, Y_te))
    return r2_folds.mean(axis=0)


def fit_subset_r2(X, Y, cols, alphas=ALPHAS, cv=CV_FOLDS):
    """Fit himalaya RidgeCV to select alpha, then compute K-fold CV R².

    Two-stage: RidgeCV selects per-target alpha on full data; K-fold CV with
    Ridge(best_alphas_) gives unbiased R² estimates. Returns (cv_r2, best_alphas).
    """
    model = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True)
    model.fit(X[:, cols], Y)
    best_alphas = model.best_alphas_
    r2 = _cv_r2_fixed_alpha(X, Y, cols, best_alphas, n_splits=cv)
    return r2, best_alphas


def compute_all_subset_r2s(X, Y, alphas=ALPHAS, cv=CV_FOLDS):
    """Fit all 7 subset models. Returns r2_dict and alpha_dict, each {name → (n_targets,)}."""
    r2_dict, alpha_dict = {}, {}
    for name, cols in SUBSETS.items():
        print(f'  Fitting subset {name}...')
        r2, best_alphas = fit_subset_r2(X, Y, cols, alphas, cv)
        r2_dict[name] = np.array(r2)
        alpha_dict[name] = np.array(best_alphas)
    return r2_dict, alpha_dict


def compute_vp_components(r2_dict):
    """
    Compute 7 Venn-diagram VP components from 7 subset R² values.

    All shapes (n_targets,). Shared components may be negative (suppression).
    Sum of all 7 components equals R²(PNM) exactly.

    Formulas (Borcard et al. 1992 inclusion-exclusion):
        unique_P   = R²(PNM) − R²(NM)
        unique_N   = R²(PNM) − R²(PM)
        unique_M   = R²(PNM) − R²(PN)
        shared_PN  = R²(PM)  + R²(NM) − R²(M)  − R²(PNM)
        shared_PM  = R²(PN)  + R²(NM) − R²(N)  − R²(PNM)
        shared_NM  = R²(PN)  + R²(PM) − R²(P)  − R²(PNM)
        shared_PNM = R²(P) + R²(N) + R²(M) − R²(PN) − R²(PM) − R²(NM) + R²(PNM)
    """
    P, N, M = r2_dict['P'], r2_dict['N'], r2_dict['M']
    PN, PM, NM, PNM = r2_dict['PN'], r2_dict['PM'], r2_dict['NM'], r2_dict['PNM']
    return {
        'unique_P':   PNM - NM,
        'unique_N':   PNM - PM,
        'unique_M':   PNM - PN,
        'shared_PN':  PM  + NM - M   - PNM,
        'shared_PM':  PN  + NM - N   - PNM,
        'shared_NM':  PN  + PM - P   - PNM,
        'shared_PNM': P + N + M - PN - PM - NM + PNM,
    }


def fit_subset_r2_fixed_alpha(X, Y, cols, best_alphas, n_splits=CV_FOLDS):
    """K-fold CV R² using Ridge with fixed per-target alpha. Wrapper around _cv_r2_fixed_alpha."""
    return _cv_r2_fixed_alpha(X, Y, cols, best_alphas, n_splits=n_splits)


def run_permutation_vp(X, Y_perms, alpha_dict):
    """
    Compute unique VP components for each phase-shifted ISPS permutation.

    Uses K-fold CV R² with fixed per-target alpha (from observed subset fits) so the
    null is properly centered. Returns {unique_P/N/M → null (n_perms, n_targets)}.
    """
    n_perms = Y_perms.shape[0]
    n_targets = Y_perms.shape[2]
    null = {k: np.empty((n_perms, n_targets))
            for k in ['unique_P', 'unique_N', 'unique_M']}

    for i in tqdm(range(n_perms), desc='Permutations'):
        Y_perm = Y_perms[i]   # (n_trs, n_targets)
        r2 = {}
        for name in PERM_SUBSETS:
            r2[name] = fit_subset_r2_fixed_alpha(X, Y_perm, SUBSETS[name], alpha_dict[name])
        null['unique_P'][i] = r2['PNM'] - r2['NM']
        null['unique_N'][i] = r2['PNM'] - r2['PM']
        null['unique_M'][i] = r2['PNM'] - r2['PN']

    return null


def compute_significance(obs, null, side='two-sided'):
    """
    Compute p-values and FDR-corrected q-values.

    obs: (n_targets,); null: (n_perms, n_targets).
    side: 'two-sided' for unique components; 'right' for full-model R²(PNM).
    Returns p_raw (n_targets,), p_fdr (n_targets,).
    """
    p_raw = p_from_null(obs, null, side=side, exact=False, axis=0)
    _, p_fdr, _, _ = multipletests(p_raw, alpha=0.05, method='fdr_bh')
    return p_raw, p_fdr


def _load_parcel_values_from_4d(vol_path, parc_path, n_parcels):
    """Read a (n_parcels, n_emotions) array from a 4D NIfTI whose values are
    constant within each parcel of `parc_path`. Mirrors how isps.py paints
    per-parcel values onto a voxel volume (one value per parcel, broadcast to
    all voxels in that parcel)."""
    vol = nib.load(vol_path).get_fdata()
    parc = nib.load(parc_path).get_fdata()
    n_emotions = vol.shape[-1]
    out = np.empty((n_parcels, n_emotions))
    for p in range(1, n_parcels + 1):
        idx = np.argwhere(parc == p)
        if idx.size == 0:
            raise ValueError(f'parcel label {p} missing from {parc_path}')
        i, j, k = idx[0]
        out[p - 1] = vol[i, j, k]
    return out


def load_isps_pvals_per_parcel(pvals_path, parc_path, n_parcels=1000):
    """Read FDR-corrected p_fdr (n_parcels, n_emotions) from
    isps_emotion_pvals_corrected.nii.gz (written by isps.py)."""
    return _load_parcel_values_from_4d(pvals_path, parc_path, n_parcels)


def load_isps_betas_per_parcel(betas_path, parc_path, n_parcels=1000):
    """Read thresholded ridge betas (n_parcels, n_emotions) from
    isps_emotion_betas_corrected.nii.gz (written by isps.py)."""
    return _load_parcel_values_from_4d(betas_path, parc_path, n_parcels)


def compute_beta_sig_mask(p_fdr_per_parcel, alpha=0.05):
    """sig_mask[p] = True iff ANY emotion's FDR p-value at parcel p is < alpha."""
    return np.any(p_fdr_per_parcel < alpha, axis=1)


def _parcels_to_nifti(data, parc_img, n_parcels):
    """Map parcel values (n_parcels, n_vols) onto a NIfTI volume."""
    parc_data = parc_img.get_fdata()
    n_vols = data.shape[1] if data.ndim == 2 else 1
    vol = np.zeros((*parc_img.shape, n_vols))
    for p in range(1, n_parcels + 1):
        mask = parc_data == p
        vol[mask, :] = data[p - 1] if data.ndim == 2 else data[p - 1]
    return nib.Nifti1Image(vol.squeeze(), parc_img.affine)


def save_vp_nifti(vp_components, p_fdr_dict, sig_mask, parc_path, out_dir,
                  n_parcels=1000):
    """Write 5 NIfTI files for VP results."""
    parc_img = nib.load(parc_path)
    emotions = ['P', 'N', 'M']

    # ISPS beta-FDR union mask  (1 volume, binary)
    nib.save(_parcels_to_nifti(sig_mask.astype(float), parc_img, n_parcels),
             os.path.join(out_dir, 'vp_isps_beta_sig_mask.nii'))

    # Raw unique R²  (3 volumes, all parcels)
    raw = np.column_stack([vp_components[f'unique_{e}'] for e in emotions])
    nib.save(_parcels_to_nifti(raw, parc_img, n_parcels),
             os.path.join(out_dir, 'vp_unique_r2_raw.nii'))

    # FDR q-values  (3 volumes; 1.0 outside sig_mask)
    pvals = np.column_stack([p_fdr_dict[f'unique_{e}'] for e in emotions])
    nib.save(_parcels_to_nifti(pvals, parc_img, n_parcels),
             os.path.join(out_dir, 'vp_pvals_corrected.nii'))

    # Thresholded unique R²  (3 volumes: nonzero only where full-model AND unique-component sig)
    thresh = np.column_stack([
        np.where(
            sig_mask & (p_fdr_dict[f'unique_{e}'] < 0.05),
            vp_components[f'unique_{e}'], 0.0
        )
        for e in emotions
    ])
    nib.save(_parcels_to_nifti(thresh, parc_img, n_parcels),
             os.path.join(out_dir, 'vp_unique_r2_thresholded.nii'))

    # All 7 components  (7 volumes, descriptive)
    all_names = ['unique_P', 'unique_N', 'unique_M',
                 'shared_PN', 'shared_PM', 'shared_NM', 'shared_PNM']
    all_comp = np.column_stack([vp_components[k] for k in all_names])
    nib.save(_parcels_to_nifti(all_comp, parc_img, n_parcels),
             os.path.join(out_dir, 'vp_all_components.nii'))

    print('Saved NIfTI files to', out_dir)


def main():
    task = 'onesmallstep'
    data_dir = f'{BASE_DIR}/outputs/{task}/data'
    parc_path = f'{BASE_DIR}/isc_scripts/schaefer_2018/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'

    print('=== Loading data ===')
    X, Y, Y_perms = load_vp_data(task)

    print('\n=== Fitting 7 observed subset models ===')
    r2_dict, alpha_dict = compute_all_subset_r2s(X, Y)

    print('\n=== Computing VP components ===')
    vp = compute_vp_components(r2_dict)
    total = sum(vp.values())
    assert np.allclose(total, r2_dict['PNM'], atol=1e-6), \
        f'VP components do not sum to R²(PNM): max diff {np.max(np.abs(total - r2_dict["PNM"])):.2e}'
    print('Sanity check passed: sum of components = R²(PNM)')
    print(f'Mean R²(PNM): {r2_dict["PNM"].mean():.4f}')
    for k, v in vp.items():
        print(f'  {k}: mean={v.mean():.4f}')

    print('\n=== Loading ISPS beta-FDR mask from isps.py outputs ===')
    pvals_path = f'{data_dir}/isps_emotion_pvals_corrected.nii.gz'
    betas_path = f'{data_dir}/isps_emotion_betas_corrected.nii.gz'
    p_fdr_per_parcel = load_isps_pvals_per_parcel(pvals_path, parc_path)
    beta_per_parcel  = load_isps_betas_per_parcel(betas_path, parc_path)
    mask_from_pvals = p_fdr_per_parcel < 0.05
    mask_from_betas = beta_per_parcel != 0.0
    assert np.array_equal(mask_from_pvals, mask_from_betas), (
        'isps_emotion_pvals_corrected.nii.gz and isps_emotion_betas_corrected.nii.gz '
        'disagree: FDR-significant p-values must coincide with nonzero betas.')
    sig_mask = compute_beta_sig_mask(p_fdr_per_parcel, alpha=0.05)
    n_full_sig = sig_mask.sum()
    print(f'  Union FDR-significant (any emotion): {n_full_sig} / {len(sig_mask)} parcels')
    for e_idx, e in enumerate(['P', 'N', 'M']):
        n_e = mask_from_pvals[:, e_idx].sum()
        print(f'    {e}: {n_e} parcels FDR-significant')

    print('\n=== Running permutation test (1024 perms) ===')
    null = run_permutation_vp(X, Y_perms, alpha_dict)

    print('\n=== Computing unique component significance (within sig_mask) ===')
    p_fdr_dict = {}
    for e in ['P', 'N', 'M']:
        key = f'unique_{e}'
        p_fdr_full = np.ones(len(sig_mask))   # non-sig parcels get p=1
        if n_full_sig > 0:
            obs_m  = vp[key][sig_mask]
            null_m = null[key][:, sig_mask]
            p_raw_m, _ = compute_significance(obs_m, null_m, side='two-sided')
            _, p_fdr_m, _, _ = multipletests(p_raw_m, alpha=0.05, method='fdr_bh')
            p_fdr_full[sig_mask] = p_fdr_m
        p_fdr_dict[key] = p_fdr_full
        n_sig = np.sum(p_fdr_full < 0.05)
        mean_r2_sig = vp[key][p_fdr_full < 0.05].mean() if n_sig > 0 else float('nan')
        print(f'  {key}: {n_sig} significant parcels (FDR<0.05 within mask), '
              f'mean unique R²={mean_r2_sig:.4f}')

    print('\n=== Saving NIfTI maps ===')
    save_vp_nifti(vp, p_fdr_dict, sig_mask, parc_path, data_dir)
    print('Done.')


if __name__ == '__main__':
    main()
