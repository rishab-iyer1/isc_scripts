"""
Helper functions to compute temporal or spatial inter-subject correlation (ISC). Requires isc_standalone.py
Temporal ISC tells us how the time courses of activity across ROIs are correlated.
Spatial ISC tells us how the coupling of neural activity changes over time.
Both can be leave-one-out (default) or pairwise. Leave-one-out compares each subject to the average of all others.

Credit to BrainIAK and their ISC tutorial found at https://brainiak.org/tutorials/10-isc/

Rishab Iyer, rsiyer@usc.edu
"""

import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
# import seaborn as sns
from nilearn.maskers import NiftiMasker

from isc_standalone import (isc)


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
    Loads BOLD data with a single ROI mask
    :param roi: name of the ROI to be loaded
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save the masked functional data for each subject
    :return: the functional file masked with the specified ROI
    """
    # all_task_names = ['onesmallstep']
    # n_subjs = {}
    # for task_name in all_task_names:
    #     n_subjs[task_name] = len(func_fns)
    #
    #
    # def load_roi_data(roi_name):
    #     # Pick a roi masker
    #     roi_masker = all_roi_masker[roi_name]
    #
    #     # Preallocate
    #     bold_roi = {task_name: [] for i, task_name in enumerate(all_task_names)}
    #
    #     # Gather data
    #     for task_name in all_task_names:
    #         for subj_id in range(n_subjs[task_name]):
    #             # Get the data for task t, subject s
    #             nii_t_s = nib.load(func_fns[subj_id])
    #             bold_roi[task_name].append(roi_masker.fit_transform(nii_t_s))
    #
    #         # Reformat the data to std form
    #         bold_roi[task_name] = np.transpose(np.array(bold_roi[task_name]), [1, 2, 0])
    #     return bold_roi
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
            print(f"subj #{n}: {subj_id} loaded from file")

    # Reformat the data to std form
    bold_roi = np.transpose(np.array(bold_roi), [1, 2, 0])

    return bold_roi


def compute_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns,
                spatial=False, pairwise=False, summary_statistic=None, tolerate_nans=True):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
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
        bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns)

        if spatial:
            bold_roi = np.transpose(bold_roi, [1, 0, 2])  # becomes (n_voxels, n_TRs, n_subjects)

        iscs_roi = isc(bold_roi, pairwise=pairwise, summary_statistic=summary_statistic, tolerate_nans=tolerate_nans)
        iscs_roi_selected[roi_name] = iscs_roi

    return iscs_roi_selected


def plot_spatial_isc(roi_selected: List[str]):
    """
    Creates a timeseries plot of spatial correlation on the y-axis vs. TRs on the isc_wholebrain-axis.
    Each ROI selected generates a new subplot in the image.
    :param roi_selected: list of all rois to compute ISC over
    :return: displays the plot
    """
    with open('../data/iscs_roi_dict.pkl', 'r') as f:
        iscs_roi_selected = pickle.load(f)
    # Plot the spatial ISC over time
    col_pal = sns.color_palette(palette='colorblind')
    ci = 95

    fig, axes = plt.subplots(len(roi_selected), 1, figsize=(14, 5 * len(roi_selected)))

    # For each ROI
    for j, roi_name in enumerate(roi_selected):
        # For each task
        sns.tsplot(
            iscs_roi_selected[roi_name],
            color=col_pal, ci=ci,
            ax=axes[j]
        )
        sns.despine()

    # Label the plot
    for j, roi_name in enumerate(roi_selected):
        axes[j].axhline(0, color='black', linestyle='--', alpha=.3)
        axes[j].set_ylabel('Linear correlation')
        axes[j].set_title('Spatial inter-subject correlation, {}'.format(roi_selected[j]))

    axes[-1].set_xlabel('TRs')

    plt.show()


def sliding_isc(roi_selected: List[str], all_roi_masker: Dict[str, NiftiMasker], func_fns, data_path: str,
                spatial=False, pairwise=False, summary_statistic='mean', tolerate_nans=True, n_trs=454,
                window_size=30, step_size=5):
    """
    Given functional data of shape (n_TRs, n_voxels, n_subjects), computes ISC for the selected ROIs.
    :param roi_selected: list of all rois to compute ISC over
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
    :param data_path: path to save loaded ROI data
    :param spatial: Whether to compute spatial ISC (default: temporal)
    :param pairwise: Whether to compute pairwise ISC (default: group)
    :param summary_statistic: Which summary statistic to use: mean or median (default: None)
    :param tolerate_nans: Whether to tolerate NaNs (default: True)
    :param n_trs: number of TRs in ISC data
    :param window_size: number of TRs in each window
    :param step_size: number of TRs to move the window by
    :return: iscs_roi_selected: a dictionary with roi name (keys) mapped to isc values (values)
    """

    from tqdm import tqdm

    # compute ISC for all ROIs
    n_windows = int((n_trs - window_size) / step_size) + 1

    iscs_roi_selected = {}
    for j, roi_name in (roi_log := tqdm(enumerate(roi_selected), leave=True)):
        roi_log.set_description(f"Computing ISC for {roi_name}")
        # Load data
        bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path)
        slide_isc = []
        for i in (window_log := tqdm(range(n_windows), leave=False)):
            window_log.set_description(f"Window {i}")
            bold_roi_window = bold_roi[i*step_size:i*step_size+window_size, :, :]
            if spatial:
                bold_roi_window = np.transpose(bold_roi, [1, 0, 2])  # becomes (n_voxels, n_TRs, n_subjects)
            iscs_roi = isc(bold_roi_window, pairwise=pairwise, summary_statistic=summary_statistic,
                           tolerate_nans=tolerate_nans)
            slide_isc.append(iscs_roi)

        iscs_roi_selected[roi_name] = np.array(slide_isc)

    return iscs_roi_selected

