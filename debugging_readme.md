# Debugging utilities for ISC-Emotion ridge analysis

This file describes quick step-by-step usage examples for the functions in
`isc_scripts/debug_utils.py`.

1) Basic validation

   - Validate ISC values (range and finiteness):

       from isc_scripts import debug_utils as dbg
       dbg.validate_isc(isc_array, name='wholebrain_isc')

   - Validate emotion consensus alignment and variance:

       dbg.validate_emotions(emotion_matrix, n_trs=454)

2) Predictor checks

   - Ensure predictors (columns of ISC matrix) have non-zero variance:

       dbg.check_predictor_variance(isc_windows)

3) Ridge regression tests

   - Quick synthetic recovery test:

       res = dbg.ridge_recovery_test()

   - Sweep alphas and inspect coefficients:

       alphas, coefs, scores = dbg.ridge_alpha_sweep(X, y)

4) Permutation testing validation

   - Create a null distribution by permuting behavior only:

       observed, nulls = dbg.validate_permutation_strategy(isc_windows, behav_windows, n_perm=1000)

   - Extract raw p-values and compare corrections:

       pvals = dbg.extract_raw_pvals(observed, nulls)
       dbg.compare_corrections(pvals)

5) Visual diagnostics

   - Histogram of p-values:

       dbg.plot_pval_histogram(pvals)

   - Plot one feature vs behavior correlation:

       dbg.plot_coef_vs_isc(isc_windows, behav_windows, feature_idx=0, emotion_idx=0)

Notes & common pitfalls:

- Use the same feature standardization and alpha grid for the real and permuted datasets.
- Run at least 1k permutations when feasible for stable p-values.
- If no voxels survive FDR, check alignment, constant predictors, alpha mismatch, and permutation strategy.
