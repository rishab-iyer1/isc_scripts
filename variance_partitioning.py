"""
Two-stage variance partitioning of ISPS by emotion consensus (P, N, M).

Stage 1 (already on disk, produced by isps.py): per-emotion FDR test of ridge
coefficients vs phase-shift null. Outputs:
  - isps_emotion_pvals_corrected.nii.gz  (3 vols: per-emotion p_fdr)
  - isps_emotion_betas_corrected.nii.gz  (3 vols: per-emotion thresholded β)

Stage 2 (this script): for each emotion E, within E's Stage-1 mask M_E only,
test whether U_E := R²(PNM) − R²(\\E) is greater than the phase-shift null,
FDR-correcting across the parcels in M_E. A parcel "supports the unique-
variance claim" for E iff it survives Stage 1 AND Stage 2.

Analysis decisions
------------------
- Alpha grid       : np.logspace(-3, 3, 20)
- Alpha selection  : RidgeCV per-target on the FULL (PNM) model only.
                     The selected per-parcel alpha is then applied to all 7
                     subset Ridge fits — observed and permuted — so that
                     U_E reflects which predictors were available, not how
                     each sub-model was regularized.
- R² metric        : Cross-validated R² (5-fold) with Ridge(alpha=best_α).
- CV folds         : 5 (alpha selection in PNM RidgeCV and outer R² CV).
- Components       : 7 Borcard inclusion-exclusion components per parcel.
                     Only unique components are permutation-tested; the
                     pairwise- and triple-shared components are descriptive
                     (can be negative under suppression).
- N permutations   : 1024 (reuses the existing phase-shift pickle).
- Stage-1 mask     : per-emotion (M_P, M_N, M_M), each = parcels with
                     p_fdr_E < 0.05 in isps_emotion_pvals_corrected.nii.gz.
                     A runtime assertion confirms the per-emotion FDR mask
                     matches the per-emotion nonzero-beta mask.
- Stage-2 FDR      : BH within M_E only. Parcels outside M_E receive p_fdr=1.
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

# All 7 subsets are now needed to compute the 6 unique-variance components
# (3 single-emotion + 3 pair):
#   U_P  = R²(PNM) - R²(NM)     U_PN = R²(PNM) - R²(M)
#   U_N  = R²(PNM) - R²(PM)     U_PM = R²(PNM) - R²(N)
#   U_M  = R²(PNM) - R²(PN)     U_NM = R²(PNM) - R²(P)
PERM_SUBSETS = ['P', 'N', 'M', 'PN', 'PM', 'NM', 'PNM']

UNIQUE_KEYS = ['U_P', 'U_N', 'U_M', 'U_PN', 'U_PM', 'U_NM']


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


def fit_pnm_alpha(X, Y, alphas=ALPHAS, cv=CV_FOLDS):
    """RidgeCV on the PNM model only. Returns per-target best_alphas (n_targets,).

    The returned per-parcel alpha is reused for every subset fit (observed and
    permuted) so that U_E = R²(PNM) − R²(\\E) is not confounded by the two
    sub-models being shrunk differently.
    """
    model = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True)
    model.fit(X[:, SUBSETS['PNM']], Y)
    return np.array(model.best_alphas_)


def compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas, cv=CV_FOLDS):
    """All 7 subsets, K-fold CV R² with Ridge(alpha=best_alphas). Returns
    {subset_name → (n_targets,)}."""
    return {name: _cv_r2_fixed_alpha(X, Y, cols, best_alphas, n_splits=cv)
            for name, cols in SUBSETS.items()}


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


def compute_unique_r2(r2_dict):
    """
    Six unique-variance components: U_S := R²(PNM) − R²(complement of S).
    Returns {key → (n_targets,)} for keys in UNIQUE_KEYS.

    For single-emotion S, the complement is the size-2 leave-one-out subset
    (so U_P = R²(PNM)−R²(NM), the unique contribution of P). For pair S, the
    complement is the size-1 left-out subset (so U_PN = R²(PNM)−R²(M), the
    variance only the {P,N} pair can explain beyond M alone).
    """
    PNM = r2_dict['PNM']
    return {
        'U_P':  PNM - r2_dict['NM'],
        'U_N':  PNM - r2_dict['PM'],
        'U_M':  PNM - r2_dict['PN'],
        'U_PN': PNM - r2_dict['M'],
        'U_PM': PNM - r2_dict['N'],
        'U_NM': PNM - r2_dict['P'],
    }


def run_permutation_vp(X, Y_perms, best_alphas):
    """
    Compute null distributions for the 6 unique-variance components.

    For each phase-shifted Y, refit all 7 subsets with Ridge(alpha=best_alphas)
    — same per-target α used for the observed fits — and form U_S for each of
    the 6 keys in UNIQUE_KEYS. Returns {key → (n_perms, n_targets)}.
    """
    n_perms = Y_perms.shape[0]
    n_targets = Y_perms.shape[2]
    null = {k: np.empty((n_perms, n_targets)) for k in UNIQUE_KEYS}

    for i in tqdm(range(n_perms), desc='Permutations'):
        Y_perm = Y_perms[i]   # (n_trs, n_targets)
        r2 = {name: _cv_r2_fixed_alpha(X, Y_perm, SUBSETS[name], best_alphas)
              for name in PERM_SUBSETS}
        unique = compute_unique_r2(r2)
        for k in UNIQUE_KEYS:
            null[k][i] = unique[k]

    return null


def compute_significance(obs, null, side='two-sided'):
    """
    Compute p-values and FDR-corrected q-values.

    obs: (n_targets,); null: (n_perms, n_targets).
    side: 'two-sided' for unique components.
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


def compute_per_emotion_masks(p_fdr_per_parcel, alpha=0.05):
    """Per-emotion Stage-1 masks: {'P','N','M' → bool mask (n_parcels,)}.

    Mask M_E[p] = True iff the FDR-corrected p-value for emotion E at parcel p
    is strictly less than alpha. No union across emotions — Stage 2 inference
    for E uses M_E directly.
    """
    return {e: p_fdr_per_parcel[:, i] < alpha
            for i, e in enumerate(['P', 'N', 'M'])}


def _parcels_to_nifti(data, parc_img, n_parcels):
    """Map parcel values (n_parcels, n_vols) onto a NIfTI volume."""
    parc_data = parc_img.get_fdata()
    n_vols = data.shape[1] if data.ndim == 2 else 1
    vol = np.zeros((*parc_img.shape, n_vols))
    for p in range(1, n_parcels + 1):
        mask = parc_data == p
        vol[mask, :] = data[p - 1] if data.ndim == 2 else data[p - 1]
    return nib.Nifti1Image(vol.squeeze(), parc_img.affine)


VP_NIFTI_MANIFEST = {
    'vp_stage1_masks.nii': {
        'units': 'binary (0 or 1)',
        'meaning': 'Stage-1 per-emotion FDR mask from isps.py (β_E ≠ 0 after FDR).',
        'volume_order': ['P', 'N', 'M'],
    },
    'vp_unique_r2_raw.nii': {
        'units': 'ΔR² (cross-validated, 5-fold)',
        'meaning': 'Raw U_S := R²(PNM) − R²(complement(S)) across all 1000 parcels. '
                   'Can be negative under CV (overfitting in reduced model).',
        'volume_order': ['U_P', 'U_N', 'U_M', 'U_PN', 'U_PM', 'U_NM'],
        'definitions': {
            'U_P':  'R²(PNM) − R²(NM)',
            'U_N':  'R²(PNM) − R²(PM)',
            'U_M':  'R²(PNM) − R²(PN)',
            'U_PN': 'R²(PNM) − R²(M)',
            'U_PM': 'R²(PNM) − R²(N)',
            'U_NM': 'R²(PNM) − R²(P)',
        },
    },
    'vp_pvals_corrected.nii': {
        'units': 'FDR-corrected q-value (BH); 1.0 outside Stage-1 mask',
        'meaning': 'Stage-2 q within each component\'s Stage-1 mask. '
                   'Single-emotion mask = isps.py FDR; pair mask = union of the two single-emotion masks.',
        'volume_order': ['U_P', 'U_N', 'U_M', 'U_PN', 'U_PM', 'U_NM'],
    },
    'vp_unique_r2_thresholded.nii': {
        'units': 'ΔR² (cross-validated, 5-fold), zero outside two-stage-significant parcels',
        'meaning': 'U_S where (Stage-1 mask AND Stage-2 q<0.05), 0 elsewhere.',
        'volume_order': ['U_P', 'U_N', 'U_M', 'U_PN', 'U_PM', 'U_NM'],
    },
    'vp_all_components.nii': {
        'units': 'R² (cross-validated, 5-fold) component value',
        'meaning': 'Borcard (1992) inclusion-exclusion partition of R²(PNM). '
                   'Sum of all 7 components per parcel equals R²(PNM). '
                   'Shared components can be negative (suppression).',
        'volume_order': ['unique_P', 'unique_N', 'unique_M',
                         'shared_PN', 'shared_PM', 'shared_NM', 'shared_PNM'],
    },
}


def save_vp_nifti(vp_components, obs_unique, p_fdr_dict, masks,
                  component_masks, parc_path, out_dir, n_parcels=1000):
    """Write NIfTI files for two-stage VP results, plus a sidecar manifest.

    See VP_NIFTI_MANIFEST for the units, meaning, and volume order of each
    output file. A JSON copy of the manifest is written to
    vp_outputs_manifest.json alongside the NIfTIs, and each NIfTI's header
    `descrip` field gets a short reminder of the volume ordering.
    """
    import json

    parc_img = nib.load(parc_path)
    emotions = ['P', 'N', 'M']

    def _write(arr, fname):
        img = _parcels_to_nifti(arr, parc_img, n_parcels)
        order = VP_NIFTI_MANIFEST[fname]['volume_order']
        # NIfTI descrip is limited to ~80 chars
        img.header['descrip'] = np.bytes_(f'vols:{",".join(order)}'[:79])
        nib.save(img, os.path.join(out_dir, fname))

    _write(np.column_stack([masks[e].astype(float) for e in emotions]),
           'vp_stage1_masks.nii')
    _write(np.column_stack([obs_unique[k] for k in UNIQUE_KEYS]),
           'vp_unique_r2_raw.nii')
    _write(np.column_stack([p_fdr_dict[k] for k in UNIQUE_KEYS]),
           'vp_pvals_corrected.nii')
    _write(np.column_stack([
        np.where(component_masks[k] & (p_fdr_dict[k] < 0.05), obs_unique[k], 0.0)
        for k in UNIQUE_KEYS
    ]), 'vp_unique_r2_thresholded.nii')
    _write(np.column_stack([vp_components[k] for k in
                            ['unique_P', 'unique_N', 'unique_M',
                             'shared_PN', 'shared_PM', 'shared_NM', 'shared_PNM']]),
           'vp_all_components.nii')

    manifest_path = os.path.join(out_dir, 'vp_outputs_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(VP_NIFTI_MANIFEST, f, indent=2)
    print(f'Saved NIfTI files + manifest to {out_dir}')


def main():
    task = 'onesmallstep'
    data_dir = f'{BASE_DIR}/outputs/{task}/data'
    parc_path = f'{BASE_DIR}/isc_scripts/schaefer_2018/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'

    print('=== Loading data ===')
    X, Y, Y_perms = load_vp_data(task)

    print('\n=== Fitting PNM RidgeCV to select per-parcel alpha ===')
    best_alphas_pnm = fit_pnm_alpha(X, Y)
    print(f'  PNM alpha distribution across parcels: '
          f'min={best_alphas_pnm.min():.3g}, median={np.median(best_alphas_pnm):.3g}, '
          f'max={best_alphas_pnm.max():.3g}')

    print('\n=== Fitting 7 observed subset models at fixed (PNM-selected) alpha ===')
    r2_dict = compute_all_subset_r2s_fixed_alpha(X, Y, best_alphas_pnm)

    print('\n=== Computing VP components ===')
    vp = compute_vp_components(r2_dict)
    total = sum(vp.values())
    assert np.allclose(total, r2_dict['PNM'], atol=1e-6), \
        f'VP components do not sum to R²(PNM): max diff {np.max(np.abs(total - r2_dict["PNM"])):.2e}'
    print('Sanity check passed: sum of components = R²(PNM)')
    print(f'Mean R²(PNM): {r2_dict["PNM"].mean():.4f}')
    for k, v in vp.items():
        print(f'  {k}: mean={v.mean():.4f}')

    print('\n=== Loading Stage-1 per-emotion masks from isps.py outputs ===')
    pvals_path = f'{data_dir}/isps_emotion_pvals_corrected.nii.gz'
    betas_path = f'{data_dir}/isps_emotion_betas_corrected.nii.gz'
    p_fdr_per_parcel = load_isps_pvals_per_parcel(pvals_path, parc_path)
    beta_per_parcel  = load_isps_betas_per_parcel(betas_path, parc_path)
    mask_from_pvals = p_fdr_per_parcel < 0.05
    mask_from_betas = beta_per_parcel != 0.0
    assert np.array_equal(mask_from_pvals, mask_from_betas), (
        'isps_emotion_pvals_corrected.nii.gz and isps_emotion_betas_corrected.nii.gz '
        'disagree: FDR-significant p-values must coincide with nonzero betas.')
    masks = compute_per_emotion_masks(p_fdr_per_parcel, alpha=0.05)
    for e in ['P', 'N', 'M']:
        print(f'  Stage-1 mask {e}: {int(masks[e].sum())} / {len(masks[e])} parcels')

    print('\n=== Computing observed unique-variance components (6 sub-models) ===')
    obs_unique = compute_unique_r2(r2_dict)
    for k in UNIQUE_KEYS:
        print(f'  {k}: mean={obs_unique[k].mean():.4f}')

    print('\n=== Building Stage-1 masks for all 6 sub-models ===')
    # Single-emotion sub-models keep their isps.py FDR mask.
    # Pair sub-models gate on the union of the two constituent single-emotion masks.
    component_masks = {
        'U_P':  masks['P'],
        'U_N':  masks['N'],
        'U_M':  masks['M'],
        'U_PN': masks['P'] | masks['N'],
        'U_PM': masks['P'] | masks['M'],
        'U_NM': masks['N'] | masks['M'],
    }
    for k in UNIQUE_KEYS:
        print(f'  Stage-1 mask {k}: {int(component_masks[k].sum())} / {len(component_masks[k])} parcels')

    print('\n=== Running permutation test (1024 perms × 7 subset fits) ===')
    null = run_permutation_vp(X, Y_perms, best_alphas_pnm)

    print('\n=== Stage 2: per-component FDR within Stage-1 mask ===')
    p_fdr_dict = {}
    for key in UNIQUE_KEYS:
        m = component_masks[key]
        p_fdr_full = np.ones(len(m))
        if m.sum() > 0:
            obs_m  = obs_unique[key][m]
            null_m = null[key][:, m]
            p_raw_m, _ = compute_significance(obs_m, null_m, side='two-sided')
            _, p_fdr_m, _, _ = multipletests(p_raw_m, alpha=0.05, method='fdr_bh')
            p_fdr_full[m] = p_fdr_m
        p_fdr_dict[key] = p_fdr_full
        both = (p_fdr_full < 0.05) & m
        n_both = int(both.sum())
        mean_in_both = obs_unique[key][both].mean() if n_both else float('nan')
        print(f'  {key}: stage1={int(m.sum())}, stage2-sig={n_both} (both stages, FDR<0.05); '
              f'mean U in both={mean_in_both:.4f}')

    print('\n=== Saving NIfTI maps + manifest + inspection pickle ===')
    save_vp_nifti(vp, obs_unique, p_fdr_dict, masks, component_masks,
                  parc_path, data_dir)

    obs_null_path = os.path.join(data_dir, 'vp_obs_null.pkl')
    pickle_payload = {
        'obs':              obs_unique,        # {key → (n_parcels,)} observed U_S, units = ΔR² CV
        'null':             null,              # {key → (n_perms, n_parcels)} phase-shift null U_S
        'masks':            masks,             # {'P','N','M' → bool (n_parcels,)} Stage-1 single-emotion masks
        'component_masks':  component_masks,   # {key → bool (n_parcels,)} Stage-1 mask used for each sub-model
        'p_fdr':            p_fdr_dict,        # {key → (n_parcels,)} Stage-2 FDR q (1.0 outside Stage-1 mask)
        'best_alphas_pnm':  best_alphas_pnm,   # (n_parcels,) per-parcel α from PNM RidgeCV
        'unique_keys':      UNIQUE_KEYS,       # canonical key ordering for tables, plots, NIfTI volumes
        'metadata': {
            'description': (
                'Two-stage variance partitioning of ISPS by emotion consensus (P, N, M). '
                'Each U_S in obs/null is the unique R² contribution of sub-model S, defined as '
                'R²(PNM) − R²(complement(S)) using 5-fold cross-validated R² with Ridge(alpha=best_alphas_pnm).'
            ),
            'units': {
                'obs':             'ΔR² (cross-validated, 5-fold); can be negative under CV',
                'null':            'ΔR² (cross-validated, 5-fold) under phase-randomized Y',
                'masks':           'bool (Stage-1: isps.py per-emotion FDR mask, q<0.05)',
                'component_masks': 'bool (Stage-1: single-emotion = M_E; pair = M_{E1} ∪ M_{E2})',
                'p_fdr':           'FDR-corrected q-value (BH); 1.0 for parcels outside Stage-1 mask',
                'best_alphas_pnm': 'Ridge α from RidgeCV on PNM (logspace(-3,3,20) grid)',
            },
            'definitions': {
                'U_P':  'R²(PNM) − R²(NM)  — unique to P',
                'U_N':  'R²(PNM) − R²(PM)  — unique to N',
                'U_M':  'R²(PNM) − R²(PN)  — unique to M',
                'U_PN': 'R²(PNM) − R²(M)   — unique to pair {P,N} beyond M',
                'U_PM': 'R²(PNM) − R²(N)   — unique to pair {P,M} beyond N',
                'U_NM': 'R²(PNM) − R²(P)   — unique to pair {N,M} beyond P',
            },
            'n_perms':         null['U_P'].shape[0],
            'n_parcels':       null['U_P'].shape[1],
            'cv_folds':        CV_FOLDS,
            'alpha_grid':      ALPHAS.tolist(),
            'stage1_alpha':    0.05,
            'stage2_alpha':    0.05,
        },
    }
    with open(obs_null_path, 'wb') as f:
        pickle.dump(pickle_payload, f)
    print(f'Saved obs/null pickle to {obs_null_path}')
    print('Done.')


if __name__ == '__main__':
    main()
