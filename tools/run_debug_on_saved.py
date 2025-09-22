#!/usr/bin/env python3
"""Runner to run quick diagnostics on saved permutation object.

This script intentionally does a light-weight check on a small subset of parcels
and permutations to validate the pipeline (ISC ranges, behavior alignment,
ridge coefficient vs null and FDR summary).
"""
import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import multipletests

# Ensure package import works when run from repo
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isc_scripts import debug_utils as dbg


def main():
    # Adjust these paths to point to your saved permutations
    perm_path = '/usr/people/ri4541/juke/isc/outputs/onesmallstep/data/sliding_isc/permutations/phaseshift_size30_step5_300parcelsparcellated_1024perms_1rois_x'
    if not os.path.exists(perm_path):
        print('Permutation file not found:', perm_path)
        return 1

    print('Loading permutation object:', perm_path)
    with open(perm_path, 'rb') as f:
        x = pickle.load(f)

    # Expect key 'wholebrain'
    if 'wholebrain' not in x:
        print('Expected key "wholebrain" in permutation object; keys:', list(x.keys()))
        return 1

    observed = x['wholebrain'][0]  # shape (n_windows, n_parcels)
    pvals_obj = x['wholebrain'][1]
    distribution = x['wholebrain'][2]  # shape (n_shifts, n_windows, n_parcels)

    print('observed shape:', observed.shape)
    print('distribution shape:', distribution.shape)

    # 1) validate isc numeric properties
    try:
        dbg.validate_isc(observed, name='observed_isc')
    except AssertionError as e:
        print('ISC validation failed:', e)
        return 1

    # 2) load behavior consensus used in ridge.py
    label_dir = '/usr/people/ri4541/juke/isc/VideoLabelling'
    coded_path = os.path.join(label_dir, 'coded_states_onesmallstep.npy')
    if not os.path.exists(coded_path):
        print('Coded states file not found:', coded_path)
        return 1

    coded_states = np.load(coded_path)
    print('coded_states shape before trimming:', coded_states.shape)
    # follow trimming in ridge.py
    coded_states = coded_states[:, :-30]
    print('coded_states shape after trimming:', coded_states.shape)

    # compute timepoint variance and sliding window mean per earlier pipeline
    n_trs = 454
    window_size = 30
    step_size = 5
    n_windows = int((n_trs - window_size) / step_size) + 1

    timepoint_variance = np.var(coded_states[:, :n_trs, :], axis=0)  # (n_trs, n_emotions)
    slide_behav = np.zeros((n_windows, timepoint_variance.shape[1]))
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        slide_behav[i] = np.mean(timepoint_variance[start_idx:end_idx], axis=0)

    # keep first 3 emotions (P, N, M) like ridge.py
    slide_behav = slide_behav[:, :3]

    dbg.validate_emotions(slide_behav, n_trs=None, name='slide_behav')

    # 3) compute ridge coefficients for a small subset of parcels
    n_parcels = observed.shape[1]
    n_check = min(12, n_parcels)
    print(f'Running RidgeCV on first {n_check} parcels (this is a light check)')

    alpha_range = np.logspace(-3, 3, 50)
    true_coefs = np.zeros((n_check, slide_behav.shape[1]))
    for p in range(n_check):
        model = RidgeCV(alphas=alpha_range, store_cv_values=True)
        model.fit(slide_behav, observed[:, p])
        true_coefs[p] = model.coef_

    # 4) compute permuted coefs for same subset using limited number of permutations
    max_perms = min(500, distribution.shape[0])
    print('Using', max_perms, 'permutations for null-check')
    perm_coefs = np.zeros((n_check, max_perms, slide_behav.shape[1]))
    for s in range(max_perms):
        shifted = distribution[s]  # shape (n_windows, n_parcels)
        for p in range(n_check):
            model = RidgeCV(alphas=alpha_range, store_cv_values=True)
            model.fit(slide_behav, shifted[:, p])
            perm_coefs[p, s] = model.coef_

    # 5) compute raw p-values and FDR across the small tested set
    from isc_scripts.debug_utils import extract_raw_pvals
    observed_small = true_coefs  # shape (n_features, n_emotions)
    # perm_coefs currently has shape (n_check, max_perms, n_emotions)
    nulls_small_re = np.transpose(perm_coefs, (1, 0, 2))  # -> (n_perm, n_features, n_emotions)

    # Debug print shapes to ensure correct orientation
    print('observed_small.shape (n_features, n_emotions):', observed_small.shape)
    print('nulls_small_re.shape (n_perm, n_features, n_emotions):', nulls_small_re.shape)

    pvals = extract_raw_pvals(observed_small, nulls_small_re)
    print('pvals shape:', pvals.shape)
    print('min raw p in tested subset per emotion:', pvals.min(axis=0))

    # FDR across tested features and emotions
    flat = pvals.flatten()
    reject, qvals, _, _ = multipletests(flat, alpha=0.05, method='fdr_bh')
    print('number significant after FDR in subset:', int(reject.sum()))

    # Quick visual outputs (optional) - print basic hist counts
    print('raw p < 0.05 counts per emotion:', (pvals < 0.05).sum(axis=0))

    return 0


if __name__ == '__main__':
    exit(main())
