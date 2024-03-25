# Import functions helpful for managing file paths
import os
from glob import glob
from os.path import join
import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.plotting import plot_stat_map
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests

from isc_standalone import (isc, bootstrap_isc, compute_summary_statistic, load_images,
                            load_boolean_mask, mask_images,
                            MaskedMultiSubjectData)

# from nilearn.input_data import NiftiMasker, NiftiLabelsMasker originally, but it's deprecated
# so replace with nilearn.maskers instead

# parser = argparse.ArgumentParser(description='Group ISC')
# parser.add_argument('--data_dir_func', type=str, description='Path to preprocessed functional data')
# parser.add_argument('--data_dir_mask', type=str, description='Path to folder with ROI masks')
# parser.add_argument('--data_dir_mni', type=str, description='Path to wholebrain mask, '
#                                                             'for example MNI152_T1_2mm_brain_mask.nii.gz')
data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data_cut/NuisanceRegressed'
data_dir_mask = '/Volumes/BCI/Ambivalent_Affect/rois'
data_dir_mni = '~/Downloads'

# Filenames for MRI data; gzipped NIfTI images (.nii.gz)
# func_fns = glob(join(data_dir, ('sub-*_task-pieman_space-MNI152NLin2009cAsym'
#                                 '_desc-tproject_bold.nii.gz')))
func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))
mask_fn = join(data_dir_mask, 'wholebrain.nii.gz')
mni_fn = join(data_dir_mni, 'MNI152_T1_2mm_brain_mask.nii.gz')

if not os.path.exists('../data/ref_nii.nii.gz'):
    # Load a NIfTI of the brain mask as a reference Nifti1Image
    ref_nii = nib.load(mask_fn)
    nib.save(ref_nii, '../data/ref_nii.nii.gz')
else:
    ref_nii = nib.load('../data/ref_nii.nii.gz')

if not os.path.exists('../data/mask_img.npy'):
    mask_img = load_boolean_mask(mask_fn)
    np.save('../data/mask_img.npy', mask_img)
else:
    mask_img = np.load('../data/mask_img.npy')


if not os.path.exists('../data/mask_coords.npy'):
    # Get coordinates of mask voxels in original image
    mask_coords = np.where(mask_img)
    np.save('../data/mask_coords.npy', mask_coords)
else:
    mask_coords = np.load('../data/mask_coords.npy')


if not os.path.exists('../data/z_scored_data.npy'):  # only runs all this code if the data isn't in current directory

    # Load functional images and masks using brainiak.io
    func_imgs = load_images(func_fns)
    # print("Using the following functional files", func_imgs)

    # Apply the brain mask using brainiak.image
    masked_imgs = mask_images(func_imgs, mask_img)

    # Collate data into a single TR isc_wholebrain voxel isc_wholebrain subject array
    data = MaskedMultiSubjectData.from_masked_images(masked_imgs, len(func_fns))

    print(f"Trimmed fMRI data shape: {data.shape} "
          f"\ni.e., {data.shape[0]} time points, {data.shape[1]} voxels, "
          f"{data.shape[2]} subjects")

    # Z-score time series for each voxel
    data = zscore(data, axis=0)

    # store z-scored data as a np file for faster reload, shouldn't run this again
    np.save('../data/z_scored_data.npy', data)
else:
    data = np.load('../data/z_scored_data.npy')


if not os.path.exists('../data/raw_isc.npy'):
    # Leave-one-out approach
    iscs = isc(data, pairwise=False, tolerate_nans=.8)
    np.save('../data/raw_isc.npy', iscs)
else:
    iscs = np.load('../data/raw_isc.npy')

# Check shape of output ISC values
print(f"ISC values shape = {iscs.shape} \ni.e., {iscs.shape[0]} "
      f"left-out subjects and {iscs.shape[1]} voxel(s)")

if not os.path.exists('../data/mean_isc.npy'):
    # Compute mean ISC (with Fisher transformation)
    mean_iscs = compute_summary_statistic(iscs, summary_statistic='mean', axis=0)
    np.save('../data/mean_isc.npy', mean_iscs)
else:
    mean_iscs = np.load('../data/mean_isc.npy')


print(f"ISC values shape = {mean_iscs.shape} \ni.e., {mean_iscs.shape[0]} "
      f"mean value across left-out subjects and {iscs.shape[1]} voxel(s)"
      f"\nMinimum mean ISC across voxels = {np.nanmin(mean_iscs):.3f}; "
      f"maximum mean ISC across voxels = {np.nanmax(mean_iscs):.3f}")

if not os.path.exists('../data/median_isc.npy'):
    # Compute median ISC
    median_iscs = compute_summary_statistic(iscs, summary_statistic='median', axis=0)
    np.save('../data/median_isc.npy', median_iscs)
else:
    median_iscs = np.load('../data/median_isc.npy')

print(f"ISC values shape = {median_iscs.shape} \ni.e., {median_iscs.shape[0]} "
      f"median value across left-out subjects and {iscs.shape[1]} voxel(s)"
      f"\nMinimum median ISC across voxels = {np.nanmin(median_iscs):.3f}; "
      f"maximum median ISC across voxels = {np.nanmax(median_iscs):.3f}")

"""
# unthresholded values based on median ISC
isc_nonthresh_median = np.full(median_iscs.shape, np.nan)
n_nans = np.sum(np.isnan(median_iscs))
print(f"{n_nans} voxels out of {median_iscs.shape[0]} are NaNs "
      f"({n_nans / median_iscs.shape[0] * 100:.2f}%)")

nonnan_mask = ~np.isnan(median_iscs)
nonnan_coords = np.where(nonnan_mask)

nonnan_isc = median_iscs[nonnan_mask]
isc_nonthresh_median[nonnan_coords] = nonnan_isc
isc_img = np.full(ref_nii.shape, np.nan)
isc_img[mask_coords] = isc_nonthresh_median
isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)

# Plot slices at coordinates -61, -20, 8
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    cut_coords=(-61, -20, 8))
# Plot slices at coordinates 0, -65, 40
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    cut_coords=(0, -65, 40))
plt.show()
# Plot slices at coordinates -61, -20, 8
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    threshold=.1,
    cut_coords=(-61, -20, 8))
# Plot slices at coordinates 0, -65, 40
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    threshold=.1,
    cut_coords=(0, -65, 40))

isc_fn = 'isc_nonthresh_median.nii.gz'
nib.save(isc_nii, isc_fn)

"""  # raw median isc without stats

"""
# unthresholded values based on mean ISC
isc_nonthresh_mean = np.full(mean_iscs.shape, np.nan)
n_nans = np.sum(np.isnan(mean_iscs))
print(f"{n_nans} voxels out of {mean_iscs.shape[0]} are NaNs "
      f"({n_nans / mean_iscs.shape[0] * 100:.2f}%)")
nonnan_mask = ~np.isnan(mean_iscs)
nonnan_coords = np.where(nonnan_mask)
nonnan_isc = mean_iscs[nonnan_mask]
isc_nonthresh_mean[nonnan_coords] = nonnan_isc
isc_img = np.full(ref_nii.shape, np.nan)
isc_img[mask_coords] = isc_nonthresh_mean
isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
# Plot slices at coordinates -61, -20, 8
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    cut_coords=(-61, -20, 8))
# Plot slices at coordinates 0, -65, 40
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    cut_coords=(0, -65, 40))
plt.show()
# Plot slices at coordinates -61, -20, 8
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    threshold=.1,
    cut_coords=(-61, -20, 8))
# Plot slices at coordinates 0, -65, 40
plot_stat_map(
    isc_nii,
    cmap='RdYlBu_r',
    vmax=.5,
    threshold=.1,
    cut_coords=(0, -65, 40))
isc_fn = 'isc_nonthresh_mean.nii.gz'
nib.save(isc_nii, isc_fn)

"""  # raw mean isc without stats

if not os.path.exists('../data/isc_thresh_pieman_n20.nii.gz'):
    # Run bootstrap hypothesis test on ISCs
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                  ci_percentile=95,
                                                  summary_statistic='median',
                                                  n_bootstraps=100000)

    # Get number of NaN voxels
    n_nans = np.sum(np.isnan(observed))
    print(f"{n_nans} voxels out of {observed.shape[0]} are NaNs "
          f"({n_nans / observed.shape[0] * 100:.2f}%)")

    # Get voxels without NaNs
    nonnan_mask = ~np.isnan(observed)
    nonnan_coords = np.where(nonnan_mask)

    # Mask both the ISC and p-value map to exclude NaNs
    nonnan_isc = observed[nonnan_mask]
    nonnan_p = p[nonnan_mask]

    # Get FDR-controlled q-values
    nonnan_q = multipletests(nonnan_p, method='fdr_bh')[1]
    threshold = .05
    print(f"{np.sum(nonnan_q < threshold)} significant voxels "
          f"controlling FDR at {threshold}")

    # Threshold ISCs according FDR-controlled threshold
    nonnan_isc[nonnan_q >= threshold] = np.nan

    # Reinsert thresholded ISCs back into whole brain image
    isc_thresh = np.full(observed.shape, np.nan)
    isc_thresh[nonnan_coords] = nonnan_isc

    # Create empty 3D image and populate
    # with thresholded ISC values
    isc_img = np.full(ref_nii.shape, np.nan)
    isc_img[mask_coords] = isc_thresh

    # Convert to NIfTI image
    isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)

    # Plot slices at coordinates -61, -20, 8
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(-61, -20, 8))

    # Plot slices at coordinates 0, -65, 40
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(0, -65, 40))
    plt.show()

    # Plot slices at coordinates -61, -20, 8
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        threshold=.1,
        cut_coords=(-61, -20, 8))

    # Plot slices at coordinates 0, -65, 40
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        threshold=.1,
        cut_coords=(0, -65, 40))

    # Save final ISC NIfTI image as .nii
    isc_fn = 'isc_thresh_pieman_n20.nii.gz'
    nib.save(isc_nii, isc_fn)
