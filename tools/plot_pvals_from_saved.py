#!/usr/bin/env python3
"""Compute raw permutation p-values from saved ISC permutations and plot maps.

This script:
 - loads saved permutation object `*_perms_*rois_x` which contains observed ISC,
   p (unused), and distribution (n_shifts, n_windows, n_parcels)
 - computes ridge coefficients for observed ISC per parcel using RidgeCV
 - groups parcels by chosen alpha, uses analytic ridge solution to compute
   coefficients for all permutations efficiently
 - computes raw p-values per parcel and emotion (no multiple-comparison correction)
 - maps p-values into the Schaefer300 parcellation and saves NIfTI + PNG plots

Designed to run quickly because permutations are already computed; does not
recompute phase-randomized ISC.
"""
import os
import sys
import numpy as np
import nibabel as nib
from collections import defaultdict
from sklearn.linear_model import RidgeCV, Ridge
import matplotlib.pyplot as plt

# ensure repository root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isc_scripts import debug_utils as dbg


def analytic_ridge_matrix(X, alpha):
    """Return matrix A such that coef = A @ y for Ridge with given alpha.
    For multi-feature design X (n_samples x n_features) solving for coefficients
    of shape (n_features,) mapping y (n_samples,) -> coef.
    A = (X^T X + alpha I)^{-1} X^T
    """
    XtX = X.T.dot(X)
    n_feat = XtX.shape[0]
    A = np.linalg.solve(XtX + alpha * np.eye(n_feat), X.T)
    return A


def map_parcel_values_to_nifti(parc_nii, n_parcels, parcel_values):
    """Map parcel values (1-indexed parcels) into voxel image using parcel map."""
    data = parc_nii.get_fdata()
    out = np.zeros(data.shape, dtype=float)
    for p in range(1, n_parcels + 1):
        mask = data == p
        if mask.sum() > 0:
            out[mask] = parcel_values[p - 1]
    return nib.Nifti1Image(out, parc_nii.affine)


def main():
    # paths (adjust if needed)
    perm_path = '/usr/people/ri4541/juke/isc/outputs/onesmallstep/data/sliding_isc/permutations/phaseshift_size30_step5_300parcelsparcellated_1024perms_1rois_x'
    parc_path = '/usr/people/ri4541/juke/isc/isc_scripts/schaefer_2018/Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
    out_dir = '/usr/people/ri4541/juke/isc/outputs/onesmallstep/figures'
    data_out_dir = '/usr/people/ri4541/juke/isc/outputs/onesmallstep/data'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(perm_path):
        print('Permutation file not found:', perm_path)
        return 1
    if not os.path.exists(parc_path):
        print('Parcellation file not found:', parc_path)
        return 1

    print('Loading permutations from', perm_path)
    import pickle
    with open(perm_path, 'rb') as f:
        x = pickle.load(f)

    key = 'wholebrain'
    if key not in x:
        print('Expected key "wholebrain" in permutation object; found:', list(x.keys()))
        return 1

    observed = x[key][0]  # (n_windows, n_parcels)
    distribution = x[key][2]  # (n_shifts, n_windows, n_parcels)
    n_shifts = distribution.shape[0]
    n_windows, n_parcels = observed.shape

    print('observed shape:', observed.shape)
    print('distribution shape:', distribution.shape)

    # compute slide_behav exactly as in ridge.py
    coded_path = '/usr/people/ri4541/juke/isc/VideoLabelling/coded_states_onesmallstep.npy'
    if not os.path.exists(coded_path):
        print('coded_states file missing:', coded_path)
        return 1
    coded_states = np.load(coded_path)
    coded_states = coded_states[:, :-30]
    n_trs = 454
    window_size = 30
    step_size = 5
    n_windows_expected = int((n_trs - window_size) / step_size) + 1
    assert n_windows == n_windows_expected, f'n_windows mismatch: {n_windows} vs {n_windows_expected}'

    timepoint_variance = np.var(coded_states[:, :n_trs, :], axis=0)
    slide_behav = np.zeros((n_windows, timepoint_variance.shape[1]))
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        slide_behav[i] = np.mean(timepoint_variance[start_idx:end_idx], axis=0)
    slide_behav = slide_behav[:, :3]

    dbg.validate_isc(observed, name='observed_isc')
    dbg.validate_emotions(slide_behav, n_trs=None, name='slide_behav')

    # Fit RidgeCV per parcel to get observed coefs and chosen alpha
    alpha_grid = np.logspace(-3, 3, 50)
    observed_coefs = np.zeros((n_parcels, slide_behav.shape[1]))
    chosen_alpha = np.zeros(n_parcels)
    print('Fitting RidgeCV per parcel to compute observed coefs and chosen alpha...')
    for p in range(n_parcels):
        y = observed[:, p]
        model = RidgeCV(alphas=alpha_grid, store_cv_values=True)
        model.fit(slide_behav, y)
        observed_coefs[p] = model.coef_
        chosen_alpha[p] = model.alpha_

    # Group parcels by alpha to vectorize permutation fits
    alpha_to_parcels = defaultdict(list)
    for idx, a in enumerate(chosen_alpha):
        alpha_to_parcels[a].append(idx)

    print('Unique alphas chosen:', len(alpha_to_parcels))

    # Precompute analytic ridge matrices for each alpha
    alpha_matrices = {}
    for a in alpha_to_parcels.keys():
        alpha_matrices[a] = analytic_ridge_matrix(slide_behav, a)

    # Compute permutation coefficients grouped by alpha
    # null_coefs shape -> (n_shifts, n_parcels, n_emotions)
    null_coefs = np.zeros((n_shifts, n_parcels, slide_behav.shape[1]))
    print('Computing coefficients across permutations...')
    for s in range(n_shifts):
        shifted = distribution[s]  # (n_windows, n_parcels)
        for a, parcels in alpha_to_parcels.items():
            A = alpha_matrices[a]  # shape (n_emotions, n_windows)
            # shifted[:, parcels] shape (n_windows, len(parcels))
            shifted_block = shifted[:, parcels]
            # coefs_block shape (n_emotions, len(parcels)) = A @ shifted_block
            coefs_block = A.dot(shifted_block)
            # place into null_coefs (transpose to (len(parcels), n_emotions))
            null_coefs[s, parcels, :] = coefs_block.T
        if (s + 1) % 100 == 0:
            print(f'  processed {s+1}/{n_shifts} permutations')

    # compute raw p-values (two-sided) per parcel and emotion
    pvals = np.zeros_like(observed_coefs)
    for p in range(n_parcels):
        for e in range(slide_behav.shape[1]):
            null = null_coefs[:, p, e]
            obs = observed_coefs[p, e]
            pvals[p, e] = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)

    print('pvals computed. min raw p per emotion:', pvals.min(axis=0))

    # Map to NIfTI and plot for each emotion
    parc_nii = nib.load(parc_path)
    for e_idx, emo in enumerate(['P', 'N', 'M']):
        pv = pvals[:, e_idx]
        # Save pval nifti
        pval_nii = map_parcel_values_to_nifti(parc_nii, n_parcels, pv)
        out_nii = os.path.join(data_out_dir, f'emotion_pvals_raw_{n_shifts}perms_{emo}.nii.gz')
        nib.save(pval_nii, out_nii)
        print('Saved p-value nifti for', emo, '->', out_nii)

        # Make an image to visualize - use -log10(p) for display
        with np.errstate(divide='ignore'):
            stat_vals = -np.log10(pv)
        stat_vals[np.isinf(stat_vals)] = np.nanmax(stat_vals[~np.isinf(stat_vals)])
        stat_nii = map_parcel_values_to_nifti(parc_nii, n_parcels, stat_vals)

        # Simple visualization using nilearn if available, otherwise save slices
        try:
            from nilearn.plotting import plot_stat_map
            from nilearn.image import new_img_like
            display = plot_stat_map(stat_nii, title=f'-log10(p) {emo}', display_mode='ortho', cut_coords=(0, 0, 0), colorbar=True)
            fig = display.frame_axes.figure
            out_png = os.path.join(out_dir, f'perm_pvals_unadjusted_{emo}.png')
            fig.savefig(out_png, dpi=150)
            display.close()
            print('Saved plot to', out_png)
        except Exception as ex:
            # fallback: save central slices as PNG
            print('nilearn not available or failed to plot:', ex)
            arr = stat_nii.get_fdata()
            mid = tuple(s // 2 for s in arr.shape)
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(np.rot90(arr[:, :, mid[2]]), cmap='hot')
            axs[0].set_title('axial')
            axs[1].imshow(np.rot90(arr[:, mid[1], :]), cmap='hot')
            axs[1].set_title('coronal')
            axs[2].imshow(np.rot90(arr[mid[0], :, :]), cmap='hot')
            axs[2].set_title('sagittal')
            plt.suptitle(f'-log10(p) {emo}')
            out_png = os.path.join(out_dir, f'perm_pvals_unadjusted_{emo}_slices.png')
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print('Saved fallback plot to', out_png)

    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
