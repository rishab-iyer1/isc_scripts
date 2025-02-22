"""
Helper functions to compute temporal or spatial inter-subject correlation (ISC). Requires isc_standalone.py
Temporal ISC tells us how the time courses of activity across ROIs are correlated. It can be visualized as a brain map where each voxel (or ROI) has a single ISC value showing how correlated it is across participants.
Spatial ISC tells us how the coupling of neural activity changes over time. It can be visualized as a time series plot (usually multiple lines where each one represents an ROI) where each time point has a corresponding ISC value.
Both can be leave-one-out (default) or pairwise. Leave-one-out compares each subject to the average of all others.

Credit to BrainIAK and their ISC tutorial found at https://brainiak.org/tutorials/10-isc/

Rishab Iyer, rsiyer@usc.edu
"""

import os
import pickle
from typing import Dict, List

# from pandas.core.common import random_state
from tqdm.auto import tqdm
import time

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from scipy.stats import pearsonr

# from ISC.scripts.Group_ISC_toystory import func_fns
# from ISC.scripts.sliding_isc import roi_selected
from isc_standalone import isc, phase_randomize, _check_timeseries_input, compute_summary_statistic, p_from_null, \
    MAX_RANDOM_SEED


def profile_function(func):
    import cProfile
    import pstats
    import io
    from functools import wraps

    @wraps(func)  # Ensures the wrapped function retains its name and docstring
    def wrapper(*args, **kwargs):
        # Initialize the profiler
        pr = cProfile.Profile()
        pr.enable()

        # Call the actual function
        result = func(*args, **kwargs)

        # Stop profiling
        pr.disable()

        # Output profiling results to a string or file
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Print the profiling results (or save them to a file)
        print(s.getvalue())  # You can replace this with writing to a file if needed

        return result

    return wrapper


def get_rois(all_roi_fpaths) -> Dict[str, NiftiMasker]:
    """
    Creates ROI masks to be used with functional data
    :param all_roi_fpaths: list of paths to all roi masks
    :return: all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    """
    # Collect all ROIs
    all_roi_names = []
    all_roi_nii = {}
    all_roi_masker = {}
    for roi_fpath in all_roi_fpaths:
        # Compute ROI name
        roi_fname = os.path.basename(roi_fpath)
        roi_name = roi_fname.split('.')[0]
        all_roi_names.append(roi_name)

        # Load roi nii file
        roi_nii = nib.load(roi_fpath)
        all_roi_nii[roi_name] = roi_nii

        # Make roi masks
        all_roi_masker[roi_name] = NiftiMasker(mask_img=roi_nii)
    return all_roi_masker


def load_roi_data(roi: str, all_roi_masker: Dict[str, NiftiMasker], func_fns: List[str], data_path: str) -> np.ndarray:
    """
    Loads BOLD data with a single ROI mask. Will create a folder within data_path to save the masked data, and if it
    already exists, will use the existing files to speed the computations up.
    :param roi: name of the ROI to be loaded
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save the masked functional data for each subject
    :return: the functional file masked with the specified ROI
    """

    subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]
    # Pick an roi masker
    roi_masker = all_roi_masker[roi]
    bold_roi = []
    # Gather data
    for n, subj_id in enumerate(subj_ids):
        # Get the data for task t, subject s
        bold_path = f"{data_path}/bold_roi/{roi}_{subj_id}.npy"
        if not os.path.exists(bold_path):
            nii_t_s = nib.load(func_fns[n])
            bold_roi.append(roi_masker.fit_transform(nii_t_s))
            print(f"subj #{n}: {subj_id} saved")
            np.save(bold_path, bold_roi[-1])
        else:
            bold_roi.append(np.load(bold_path))
            print(f"ROI {roi}, subj #{n}: {subj_id} loaded from file")

    assert all([bold_roi[0].shape == bold_roi[i].shape for i in
                range(1, len(bold_roi))]), "dimensions are not consistent"  # check that all the dimensions are the same

    # Reformat the data from (n_subjects, n_TRs, n_voxels) to (n_TRs, n_voxels, n_subjects) to prepare for ISC
    bold_roi = np.transpose(np.array(bold_roi), [1, 2, 0])

    return bold_roi


def compute_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns, data_path: str,
                spatial=False, pairwise=False, summary_statistic=None, tolerate_nans=True):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save the masked functional data for each subject
    :param spatial: Whether to compute spatial ISC (default: temporal)
    :param pairwise: Whether to compute pairwise ISC (default: group)
    :param summary_statistic: Which summary statistic to use: mean or median (default: None)
    :param tolerate_nans: Whether to tolerate NaNs (default: True)
    :return: iscs_roi_selected: a dictionary with roi name (keys) mapped to isc values (values)
    """
    # compute ISC for all ROIs
    iscs_roi_selected = {}
    for j, roi_name in enumerate(roi_selected):
        print(j, roi_name)

        # Load data
        bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path=data_path)

        if spatial:
            bold_roi = np.transpose(bold_roi, [1, 0, 2])  # becomes (n_voxels, n_TRs, n_subjects)

        iscs_roi = isc(bold_roi, pairwise=pairwise, summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)
        iscs_roi_selected[roi_name] = iscs_roi

    return iscs_roi_selected


# def plot_spatial_isc(roi_selected: List[str]):
#     """
#     Creates a timeseries plot of spatial correlation on the y-axis vs. TRs on the x-axis.
#     Each ROI selected generates a new subplot in the image.
#     :param roi_selected: list of all rois to compute ISC over
#     :return: displays the plot
#     """
#     with open('../data/iscs_roi_dict.pkl', 'r') as f:
#         iscs_roi_selected = pickle.load(f)
#     # Plot the spatial ISC over time
#     col_pal = sns.color_palette(palette='colorblind')
#     ci = 95
#
#     fig, axes = plt.subplots(len(roi_selected), 1, figsize=(14, 5 * len(roi_selected)))
#
#     # For each ROI
#     for j, roi_name in enumerate(roi_selected):
#         # For each task
#         sns.tsplot(
#             iscs_roi_selected[roi_name],
#             color=col_pal, ci=ci,
#             ax=axes[j]
#         )
#         sns.despine()
#
#     # Label the plot
#     for j, roi_name in enumerate(roi_selected):
#         axes[j].axhline(0, color='black', linestyle='--', alpha=.3)
#         axes[j].set_ylabel('Linear correlation')
#         axes[j].set_title('Spatial inter-subject correlation, {}'.format(roi_selected[j]))
#
#     axes[-1].set_xlabel('TRs')
#
#     plt.show()


def sliding_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns, n_trs: int, data_path: str,
                avg_over_roi=True, spatial=False, pairwise=False,
                summary_statistic='median', tolerate_nans=True, window_size=30, step_size=5):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes sliding window ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save loaded ROI data
    :param avg_over_roi: whether to average the time series over the ROI before computing ISC
    :param spatial: whether to compute spatial ISC (default: temporal)
    :param pairwise: whether to compute pairwise ISC (default: group)
    :param summary_statistic: Which summary statistic to use: mean or median (default: None)
    :param tolerate_nans: Whether to tolerate NaNs (default: True)
    :param n_trs: number of TRs in ISC data
    :param window_size: number of TRs in each window
    :param step_size: number of TRs to move the window by
    :return: iscs_roi_selected: a dictionary with roi name (keys) mapped to isc values (values)
    """
    iscs_roi_selected = {}
    for j, roi_name in (roi_log := tqdm(enumerate(roi_selected), leave=True, position=0, ascii=True)):
        roi_log.set_description(f"Computing ISC for {roi_name}")
        # Load data
        bold_roi = load_roi_data(roi=roi_name, all_roi_masker=all_roi_masker, func_fns=func_fns,
                                 data_path=data_path)  # shape (n_TRs, n_voxels, n_subjects)

        iscs_roi_selected[roi_name] = _compute_sliding_isc(bold_roi, n_trs=n_trs, window_size=window_size,
                                                           step_size=step_size, avg_over_roi=avg_over_roi,
                                                           spatial=spatial, pairwise=pairwise,
                                                           summary_statistic=summary_statistic,
                                                           tolerate_nans=tolerate_nans)
    return iscs_roi_selected


def phaseshift_sliding_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns, n_trs: int,
                           data_path: str, avg_over_roi=True, spatial=False, pairwise=False,
                           summary_statistic='median', tolerate_nans=True, random_state=None, n_shifts=1000, window_size=30, step_size=5):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes sliding window ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save loaded ROI data
    :param avg_over_roi: whether to average the time series over the ROI before computing ISC
    :param spatial: whether to compute spatial ISC (default: temporal)
    :param pairwise: whether to compute pairwise ISC (default: group)
    :param summary_statistic: Which summary statistic to use: mean or median (default: None)
    :param tolerate_nans: Whether to tolerate NaNs (default: True)
    :param random_state:
    :param n_shifts:
    :param n_trs: number of TRs in ISC data
    :param window_size: number of TRs in each window
    :param step_size: number of TRs to move the window by
    :return: iscs_roi_selected: a dictionary with roi name (keys) mapped to isc values (values)
    """
    iscs_roi_selected = {}
    for j, roi_name in (roi_log := tqdm(enumerate(roi_selected), leave=True, position=0, ascii=True)):
        roi_log.set_description(f"Computing ISC for {roi_name}")
        # Load data
        bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path)  # shape (n_TRs, n_voxels, n_subjects)

        iscs_roi_selected[roi_name] = _compute_phaseshift_sliding_isc(bold_roi, n_trs=n_trs, window_size=window_size,
                                                                      step_size=step_size,
                                                                      avg_over_roi=avg_over_roi, spatial=spatial,
                                                                      pairwise=pairwise,
                                                                      summary_statistic=summary_statistic,
                                                                      n_shifts=n_shifts,
                                                                      tolerate_nans=tolerate_nans,
                                                                      random_state=random_state)
    return iscs_roi_selected


def phaseshift_sliding_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns, n_trs: int,
                           data_path: str, avg_over_roi=True, spatial=False, pairwise=False,
                           summary_statistic='median', tolerate_nans=True, random_state=None, n_shifts=1000, window_size=30, step_size=5):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes sliding window ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save loaded ROI data
    :param avg_over_roi: whether to average the time series over the ROI before computing ISC
    :param spatial: whether to compute spatial ISC (default: temporal)
    :param pairwise: whether to compute pairwise ISC (default: group)
    :param summary_statistic: Which summary statistic to use: mean or median (default: None)
    :param tolerate_nans: Whether to tolerate NaNs (default: True)
    :param random_state:
    :param n_shifts:
    :param n_trs: number of TRs in ISC data
    :param window_size: number of TRs in each window
    :param step_size: number of TRs to move the window by
    :return: iscs_roi_selected: a dictionary with roi name (keys) mapped to isc values (values)
    """
    iscs_roi_selected = {}
    for j, roi_name in (roi_log := tqdm(enumerate(roi_selected), leave=True, position=0, ascii=True)):
        roi_log.set_description(f"Computing ISC for {roi_name}")
        # Load data
        bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path)  # shape (n_TRs, n_voxels, n_subjects)

        iscs_roi_selected[roi_name] = _compute_phaseshift_sliding_isc(bold_roi, n_trs=n_trs, window_size=window_size,
                                                                      step_size=step_size,
                                                                      avg_over_roi=avg_over_roi, spatial=spatial,
                                                                      pairwise=pairwise,
                                                                      summary_statistic=summary_statistic,
                                                                      n_shifts=n_shifts,
                                                                      tolerate_nans=tolerate_nans,
                                                                      random_state=random_state)
    return iscs_roi_selected



def _permutation_task(i, data, n_trs, window_size, step_size, avg_over_roi, spatial,
                      pairwise, summary_statistic, tolerate_nans, random_state):
    """Helper function to handle the permutation task."""
    # print('Using process', os.getpid())
    prng = np.random.RandomState(random_state)
    shifted_data = phase_randomize(data, random_state=prng)

    if pairwise:
        shifted_isc = _compute_sliding_isc(shifted_data, n_trs, window_size, step_size, avg_over_roi,
                                           spatial=spatial, pairwise=pairwise, summary_statistic=summary_statistic,
                                           tolerate_nans=tolerate_nans)
    else:
        shifted_data = np.rollaxis(shifted_data, 2, 0)
        shifted_isc = []
        for s, shifted_subject in enumerate(shifted_data):
            nonshifted_mean = np.mean(np.delete(data, s, axis=2), axis=2)
            loo_isc = _compute_sliding_isc(np.dstack((shifted_subject, nonshifted_mean)), n_trs, window_size,
                                           step_size, avg_over_roi, spatial=spatial, pairwise=pairwise,
                                           summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)
            shifted_isc.append(loo_isc)

        shifted_isc = compute_summary_statistic(np.dstack(shifted_isc), summary_statistic=summary_statistic, axis=2)

    return shifted_isc


def _compute_phaseshift_sliding_isc_parallel(data, n_trs, window_size, step_size, avg_over_roi=True, spatial=False,
                                    pairwise=False, summary_statistic='median', n_shifts=1000, tolerate_nans=True,
                                    random_state=None, n_jobs=None):
    """Phase randomization for one-sample ISC test with optional parallelization"""

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    from tqdm import tqdm

    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)
    observed = _compute_sliding_isc(data, n_trs, window_size, step_size, avg_over_roi, spatial=spatial,
                                    pairwise=pairwise, summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)

    # Create a random state if needed
    if random_state is None:
        random_state = np.random.RandomState()

    distribution = []

    with tqdm(total=n_shifts) as pbar:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(_permutation_task, i, data, n_trs, window_size, step_size,
                                avg_over_roi, spatial, pairwise, summary_statistic, tolerate_nans,
                                random_state.randint(0, MAX_RANDOM_SEED))
                for i in range(n_shifts)
            ]

            for future in as_completed(futures):
                pbar.update(1)
                distribution.append(future.result())

    distribution = np.stack(distribution)

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, distribution, side='two-sided', exact=False, axis=0)

    return observed, p, distribution

# @profile_function
def _compute_sliding_isc(data, n_trs, window_size, step_size, avg_over_roi=True, spatial=False, pairwise=False,
                         summary_statistic='median', tolerate_nans=True):
    """
    :param data:
    :param n_trs:
    :param window_size:
    :param step_size:
    :param avg_over_roi:
    :param spatial:
    :param pairwise:
    :param summary_statistic:
    :param tolerate_nans:
    :return:
    """
    n_windows = int((n_trs - window_size) / step_size) + 1
    slide_isc = []

    for i in range(n_windows):
        data_window = data[i * step_size:i * step_size + window_size, :, :]  # shape (window_size, n_voxels, n_subjects)
        if avg_over_roi:
            data_window = np.mean(data_window, axis=1, keepdims=True)  # shape (window_size, n_voxels=1, n_subjects)
        if spatial:
            data_window = np.transpose(data_window, [1, 0, 2])  # becomes (n_voxels, window_size, n_subjects)
        slide_isc.append(isc(data_window, pairwise=pairwise, summary_statistic=summary_statistic, tolerate_nans=tolerate_nans))
    return np.array(slide_isc)

# @profile_function
def _compute_phaseshift_sliding_isc(data, n_trs, window_size, step_size, avg_over_roi=True, spatial=False,
                                    pairwise=False,
                                    summary_statistic='median', n_shifts=1000, tolerate_nans=True, random_state=None):
    """Phase randomization for one-sample ISC test
    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISC

    n_trs :
    window_size :
    step_size :
    avg_over_roi :
    spatial :
    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_shifts : int, default: 1000
        Number of randomly shifted samples

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    random_state : int, None, or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs

    p : float, p-value
        p-value based on time-shifting randomization test

    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """
    # start = time.perf_counter()
    # print('using process', os.getpid())
    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Get actual observed ISC
    observed = _compute_sliding_isc(data, n_trs, window_size, step_size, avg_over_roi, spatial=spatial,
                                    pairwise=pairwise, summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)

    # Iterate through randomized shifts to create null distribution
    distribution = []
    # for i in tqdm(np.arange(n_shifts), position=0, leave=True):
    for i in np.arange(n_shifts):
        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get shifted version of data
        shifted_data = phase_randomize(data, random_state=prng)
        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = _compute_sliding_isc(shifted_data, n_trs, window_size, step_size, avg_over_roi,
                                               spatial=spatial, pairwise=pairwise, summary_statistic=summary_statistic,
                                               tolerate_nans=tolerate_nans)

        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:

            # Roll subject axis of phase-randomized data
            shifted_data = np.rollaxis(shifted_data, 2, 0)

            shifted_isc = []
            for s, shifted_subject in enumerate(shifted_data):
                # ISC of shifted left-out subject vs mean of N-1 subjects
                nonshifted_mean = np.mean(np.delete(data, s, axis=2),
                                          axis=2)
                # print(time.time() - end)
                loo_isc = _compute_sliding_isc(np.dstack((shifted_subject, nonshifted_mean)), n_trs, window_size,
                                               step_size, avg_over_roi, spatial=spatial, pairwise=pairwise,
                                               summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)
                shifted_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            shifted_isc = compute_summary_statistic(
                np.dstack(shifted_isc),
                summary_statistic=summary_statistic, axis=2)
        distribution.append(shifted_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.stack(distribution)

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, distribution,
                    side='two-sided', exact=False,
                    axis=0)
    # print('completed in', time.perf_counter() - start)
    return observed, p, distribution


def permute_isc_behav(isc_data: np.ndarray, behav: np.ndarray, n_perm: int,
                      voxel_idx: int, perm_path: str) -> np.ndarray:
    """
    Perform permutation testing comparing intersubject correlation (ISC) with some behavioral measure to test the null
    hypothesis that there is no correlation between ISC at a given voxel and the behavioral measure. Shuffles the
    behavioral report and computes the correlation between the shuffled report and the ISC. This is done many times
    to create a null distribution of correlations. The p-value is then computed as the proportion of the null
    distribution that is greater than the observed correlation.

    :param isc_data: ISC data
    :param behav: behavioral data
    :param n_perm: number of permutations to compute
    :param voxel_idx: permutations are done on a single voxel assuming the null distribution is the same for all voxels
    :param perm_path: path to save permutation results
    :return: saves the permutation results to a pickle file
    """
    vox_idx = voxel_idx  # pick just one voxel to do permutations on
    perm = np.empty(shape=(behav.shape[1], n_perm, 2))  # number of emotions, n_perm, r and p
    if not os.path.exists(perm_path):  # only compute if file DNE
        rng = np.random.default_rng()  # for rng.permutation
        for e in tqdm(range(behav.shape[1]), leave=True):  # number of emotions
            for i in tqdm(range(n_perm), leave=False):  # number of permutations
                nan_mask = ~np.isnan(behav[:, e])  # mask to ignore nans for any given pair
                perm[e, i] = pearsonr(isc_data.T[vox_idx][nan_mask],
                                      rng.permutation(behav[:, e][nan_mask]))
        # save perm to pickle
        with open(perm_path, 'wb') as f:
            pickle.dump(perm, f)
    else:
        with open(perm_path, 'rb') as f:
            perm = pickle.load(f)
        print("Permutation results loaded from file.")

    return perm


def print_stats(data):
    print("NaN count in data:", np.isnan(data).sum())
    print("Inf count in data:", np.isinf(data).sum())
    print("Max value in data:", np.nanmax(data))
    print("Min value in data:", np.nanmin(data))
    print("Shape:", data.shape)


def parcellate_bold(data, n_parcels, masked_parc):
    # data = [x for x in bold_roi]
    # data = data[0]
    print(data.shape)
    data = np.nan_to_num(data, nan=0.0)  # replace nan with 0
    data = np.clip(data, -1e6, 1e6)  # clip max and min values
    from scipy.stats import zscore
    # data = zscore(data, axis=0)
    
    all_parcel_data = []
    # n_parcels = 1000
    # Initialize output (timepoints, parcels, subjects)
    parcel_ts = np.zeros((data.shape[0], n_parcels, data.shape[2]))
    
    # Loop over each subject independently
    for subj_idx in range(data.shape[2]):
        for parcel_id in range(1, n_parcels + 1):
            parcel_voxels = np.where(masked_parc == parcel_id)[0]
            if parcel_voxels.size > 0:
                parcel_ts[:, parcel_id - 1, subj_idx] = np.mean(data[:, parcel_voxels, subj_idx], axis=1)
    
    all_parcel_data.append(parcel_ts)  # Shape: (454, 1000, 27)
    
    all_parcel_data = np.array(all_parcel_data)[0]  # Shape: (num_subjects, 454, 1000, 27)
    print_stats(all_parcel_data)
    all_parcel_data = zscore(all_parcel_data, axis=0)
    print_stats(all_parcel_data)
    return all_parcel_data

def load_schaeffer1000(parc_path, mask_path):
    parc = nib.load(parc_path)
    mask = np.load(mask_path)
    assert np.all(parc.shape == mask.shape)
    masked_parc = parc.get_fdata().flatten()[mask.flatten()]
    return parc, masked_parc


def parcel_to_nifti(parc, n_parcels, input_data, saving=False, save_path=None):
    """
    Given an np array, convert to a nifti map. Useful for visualizing data as a nifti file. 
    :param parc: parcellation as a nifti object
    """
    if saving: 
        assert save_path is not None, "No save path provided"

    img = np.zeros(parc.shape)
    for p in range(1, n_parcels + 1):
        mask = parc.get_fdata() == p  # location of current parcel
        img[mask, :] = input_data[:, p - 1].T

    nifti = nib.Nifti1Image(img, parc.affine)
    if saving:
        nib.save(nifti, save_path)
    return nifti

def main():
    pass


if __name__ == '__main__':
    main()
