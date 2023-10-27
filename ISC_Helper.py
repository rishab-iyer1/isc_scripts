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


def load_roi_data(roi: str, all_roi_masker: Dict[str, NiftiMasker], func_fns: List[str]) -> np.ndarray:
    """
    Loads BOLD data with a single ROI mask
    :param roi: name of the ROI to be loaded
    :param all_roi_masker: a dictionary with roi name (keys) mapped to NiftiMasker object (values)
    :param func_fns: file names of all functional data
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

    n_subjs = len(func_fns)
    # Pick an roi masker
    roi_masker = all_roi_masker[roi]
    bold_roi = []
    # Gather data
    for subj_id in range(n_subjs):
        # Get the data for task t, subject s
        nii_t_s = nib.load(func_fns[subj_id])
        bold_roi.append(roi_masker.fit_transform(nii_t_s))
        print(f"subj #{subj_id} loaded and transformed")

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
