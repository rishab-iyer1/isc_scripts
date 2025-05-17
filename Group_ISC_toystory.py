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
from tqdm import tqdm

from isc_standalone import (isc, bootstrap_isc, compute_summary_statistic, load_images,
                            load_boolean_mask, mask_images,
                            MaskedMultiSubjectData)

# from nilearn.input_data import NiftiMasker, NiftiLabelsMasker originally, but it's deprecated
# so replace with nilearn.maskers instead

# Add options to compute raw (unthresholded) maps for mean and median ISC
parser = argparse.ArgumentParser(description='Group ISC')
parser.add_argument('--compute_raw_mean', action='store_true', help='Compute raw unthresholded mean ISC map')
parser.add_argument('--compute_raw_median', action='store_true', help='Compute raw unthresholded median ISC map')
parser.add_argument('--plot_raw_mean', action='store_true', help='Plot raw unthresholded mean ISC map')
parser.add_argument('--plot_raw_median', action='store_true', help='Plot raw unthresholded median ISC map')
args = parser.parse_args()

# Define directories
data_dir_func = '/jukebox/norman/rsiyer/isc/toystory/nuisance_regressed_cut'
data_dir_mask = '/jukebox/norman/rsiyer/isc/isc_scripts/rois'
data_dir_mni = '/jukebox/norman/rsiyer/isc/toystory'
output_dir = '/jukebox/norman/rsiyer/isc/outputs/toystory/data'
os.makedirs(output_dir, exist_ok=True)

# Define file paths
func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))
mask_fn = join(data_dir_mask, 'wholebrain.nii.gz')
mni_fn = join(data_dir_mni, 'MNI152_T1_2mm_brain_mask.nii.gz')
ref_nii = nib.load(mask_fn)

# Load mask
mask_img_path = join(output_dir, 'mask_img.npy')
if not os.path.exists(mask_img_path):
    mask_img = load_boolean_mask(mask_fn)
    np.save(mask_img_path, mask_img)
else:
    mask_img = np.load(mask_img_path)

# Load mask coords
mask_coords_path = join(output_dir, 'mask_coords.npy')
if not os.path.exists(mask_coords_path):
    mask_coords = np.where(mask_img)
    np.save(mask_coords_path, mask_coords)
else:
    mask_coords = np.load(mask_coords_path)

# Load or preprocess data
data_path = join(output_dir, 'z_scored_data.npy')
if not os.path.exists(data_path):
    print("Loading functional images...")
    func_imgs = load_images(func_fns)

    print("Applying mask to functional images...")
    masked_imgs = mask_images(func_imgs, mask_img)

    print("Creating MaskedMultiSubjectData object and z-scoring data...")
    data = MaskedMultiSubjectData.from_masked_images(masked_imgs, len(func_fns))
    data = zscore(data, axis=0)

    print("Saving z-scored data...")
    np.save(data_path, data)
else:
    data = np.load(data_path)

# Compute raw ISC
isc_path = join(output_dir, 'raw_isc.npy')
if not os.path.exists(isc_path):
    print("Computing raw ISC values...")
    iscs = isc(data, pairwise=False, tolerate_nans=0.8)
    np.save(isc_path, iscs)
else:
    iscs = np.load(isc_path)

# Compute mean ISC
mean_isc_path = join(output_dir, 'mean_isc.npy')
if not os.path.exists(mean_isc_path):
    print("Computing mean ISC...")
    mean_iscs = compute_summary_statistic(iscs, summary_statistic='mean', axis=0)
    np.save(mean_isc_path, mean_iscs)
    
else:
    mean_iscs = np.load(mean_isc_path)


# Compute raw unthresholded mean ISC map if requested
if args.compute_raw_mean:
    print("Computing raw unthresholded mean ISC map...")
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
    isc_fn = join(output_dir, 'isc_nonthresh_mean.nii.gz')
    nib.save(isc_nii, isc_fn)
    print(f"ISC values shape = {mean_iscs.shape} \ni.e., {mean_iscs.shape[0]} "
            f"mean value across left-out subjects and {iscs.shape[1]} voxel(s)"
            f"\nMinimum mean ISC across voxels = {np.nanmin(mean_iscs):.3f}; "
            f"maximum mean ISC across voxels = {np.nanmax(mean_iscs):.3f}")
    

# Plot raw unthresholded mean ISC map if requested
if args.plot_raw_mean:
    print("Plotting raw unthresholded mean ISC map...")
    isc_img = np.full(ref_nii.shape, np.nan)
    isc_img[mask_coords] = mean_iscs
    isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(-61, -20, 8))
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(0, -65, 40))
    plt.show()


# Compute median ISC
median_isc_path = join(output_dir, 'median_isc.npy')
if not os.path.exists(median_isc_path):
    print("Computing median ISC...")
    median_iscs = compute_summary_statistic(iscs, summary_statistic='median', axis=0)
    np.save(median_isc_path, median_iscs)
else:
    median_iscs = np.load(median_isc_path)


print(f"ISC values shape = {median_iscs.shape} \ni.e., {median_iscs.shape[0]} "
      f"median value across left-out subjects and {iscs.shape[1]} voxel(s)"
      f"\nMinimum median ISC across voxels = {np.nanmin(median_iscs):.3f}; "
      f"maximum median ISC across voxels = {np.nanmax(median_iscs):.3f}")


# """
# # unthresholded values based on median ISC
# isc_nonthresh_median = np.full(median_iscs.shape, np.nan)
# n_nans = np.sum(np.isnan(median_iscs))
# print(f"{n_nans} voxels out of {median_iscs.shape[0]} are NaNs "
#       f"({n_nans / median_iscs.shape[0] * 100:.2f}%)")

# nonnan_mask = ~np.isnan(median_iscs)
# nonnan_coords = np.where(nonnan_mask)

# nonnan_isc = median_iscs[nonnan_mask]
# isc_nonthresh_median[nonnan_coords] = nonnan_isc
# isc_img = np.full(ref_nii.shape, np.nan)
# isc_img[mask_coords] = isc_nonthresh_median
# isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)

# Create full-brain ISC image and populate masked voxels
isc_img = np.full(ref_nii.shape, np.nan)
isc_img[tuple(mask_coords)] = median_iscs

# Convert to NIfTI
isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)

# Optional: Save or plot
nib.save(isc_nii, join(output_dir, 'isc_median.nii.gz'))

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


# Compute raw unthresholded median ISC map if requested
if args.compute_raw_median:
    print("Computing raw unthresholded median ISC map...")
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
    isc_fn = join(output_dir, 'isc_nonthresh_median.nii.gz')
    nib.save(isc_nii, isc_fn)


# Plot raw unthresholded median ISC map if requested
if args.plot_raw_median:
    print("Plotting raw unthresholded median ISC map...")
    isc_img = np.full(ref_nii.shape, np.nan)
    isc_img[mask_coords] = median_iscs
    isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(-61, -20, 8))
    plot_stat_map(
        isc_nii,
        cmap='RdYlBu_r',
        vmax=.5,
        cut_coords=(0, -65, 40))
    plt.show()

n_bootstraps = 10
save_path = join(output_dir, f'isc_thresh_{n_bootstraps}bootstraps.nii.gz')
if not os.path.exists(save_path):
    # Run bootstrap hypothesis test on ISCs
    observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                  ci_percentile=95,
                                                  summary_statistic='median',
                                                  n_bootstraps=n_bootstraps)

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
    nib.save(isc_nii, save_path)
