"""
This script computes pairwise temporal ISC for a set of ROIs and compares it to pairwise behavioral ratings.
"""
import itertools
import os
import pickle
from copy import deepcopy
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import xarray as xr
from nilearn.glm import fdr_threshold
from scipy.stats import pearsonr, norm
from tqdm import tqdm

from ISC_Helper import compute_isc, get_rois

# -------------------------------
# File paths
# -------------------------------
data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data_cut/NuisanceRegressed'
func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))
roi_mask_path = '/Volumes/BCI/Ambivalent_Affect/rois'
all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii.gz'))
data_path = '/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data'
rating_path = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/coded_df.nc'

# -------------------------------
# Parameters
# -------------------------------
subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]
subj_ids.sort()
roi_selected = ['auditory', 'ACC', 'vmPFC', 'insula', 'visualcortex', 'amygdala', 'wholebrain']  # ['auditory',
# 'visualcortex', 'ACC', 'vmPFC', 'vPCUN', 'aINS_L', 'aANG_L', 'pANG_L', 'Insular_R', 'dPCUN', 'aANG_R', 'aCUN',
# 'pANG_R', 'PMC_L', 'dPCC', 'insula', 'amygdala', 'wholebrain']
emotions = ['P', 'N', 'M', 'X', 'Cry']
spatial = False
pairwise = True

# -------------------------------
# Compute and save ISC
# -------------------------------
all_roi_masker = get_rois(all_roi_fpaths)
spatial_name = "spatial" if spatial else "temporal"
pairwise_name = "pairwise" if pairwise else "group"
isc_path = f"{data_path}/isc_{spatial_name}_{pairwise_name}_n{len(subj_ids)}_roi{len(roi_selected)}.pkl"

if not os.path.exists(isc_path):
    iscs_roi_selected = compute_isc(roi_selected, all_roi_masker, func_fns, spatial=spatial, pairwise=pairwise)
    with open(isc_path, 'wb') as f:
        pickle.dump(iscs_roi_selected, f)
else:
    with open(isc_path, 'rb') as f:
        iscs_roi_selected = pickle.load(f)


# create a mapping to keep track of the subjects we compared pairwise
subj_mapping = list()
for i, (s1, s2) in enumerate(itertools.combinations(subj_ids, 2)):
    subj_mapping.append((s1, s2))

# -------------------------------
# Behavioral ratings
# -------------------------------
df = xr.open_dataset(rating_path)
df = df.sel(subj_id=subj_ids)  # subset df with only the subjects we have ISC for

n_emo = 5
df_emotions = ['P', 'N', 'M', 'X', 'Cry', 'P_smooth', 'N_smooth', 'M_smooth', 'X_smooth', 'Cry_smooth']
df_consensus = np.empty(shape=(len(subj_mapping), 5))
for i, (s1, s2) in enumerate(subj_mapping):
    for j, emo in enumerate(df_emotions[5:]):
        df_consensus[i, j] = pearsonr(df.sel(subj_id=s1, emotion=emo).to_array()[0, :],
                                      df.sel(subj_id=s2, emotion=emo).to_array()[0, :])[0]
# list the number of nans in each column
print(np.sum(np.isnan(df_consensus), axis=0))
n_perm = 1000000
alpha = int(n_perm * 0.05)
isc_wholebrain = iscs_roi_selected['wholebrain']


def permute_isc_behav(isc: np.ndarray, behav: np.ndarray, num_perms: int, voxel_idx: int, perm_path: str):
    """
    Perform permutation testing comparing intersubject correlation (ISC) with some behavioral measure to test the null
    hypothesis that there is no correlation between ISC at a given voxel and the behavioral measure. Shuffles the
    behavioral report and computes the correlation between the shuffled report and the ISC. This is done many times
    to create a null distribution of correlations. The p-value is then computed as the proportion of the null
    distribution that is greater than the observed correlation.

    :param isc: ISC data
    :param behav: behavioral data
    :param num_perms: number of permutations
    :param voxel_idx: permutations are done on a single voxel assuming the null distribution is the same for all voxels
    :return: saves the permutation results to a pickle file
    """
    vox_idx = voxel_idx  # pick just one voxel to do permutations on
    perm_vox = np.empty(shape=(behav.shape[1], num_perms, 2))  # number of emotions, n_perm, r and p
    if not os.path.exists(perm_path):  # only compute if file DNE
        rng = np.random.default_rng()  # for rng.permutation
        for e in tqdm(range(df_consensus.shape[1])):  # number of emotions
            for i in tqdm(range(n_perm)):  # number of permutations
                nan_mask = ~np.isnan(df_consensus[:, e])  # mask to ignore nans for any given pair
                perm_vox[e, i] = pearsonr(isc.T[vox_idx][nan_mask],
                                          rng.permutation(df_consensus[:, e][nan_mask]))
        # save perm to pickle
        with open(perm_path, 'wb') as f:
            pickle.dump(perm_vox, f)
    else:
        with open(perm_path, 'rb') as f:
            perm_vox = pickle.load(f)

perm_vox = permute_isc_behav(isc_wholebrain, df_consensus, n_perm, 1000, f"{data_path}/perm_vox_{n_perm}.pkl")

# view histogram
plt.hist(perm_vox[4, :, 0], bins=100)
plt.title('Histogram of permuted correlations for cry emotion in one voxel')
plt.show()

# print critical values for each emotion based on permutation testing
for e in range(5):
    print(f"Critical value for {df_emotions[e+5]} at p=0.05: {np.sort(perm_vox[e, :, 0])[-500]:.3f}")

s_map = np.empty(shape=(n_emo, iscs_roi_selected['wholebrain'].shape[1], 2))
if not os.path.exists(f"{data_path}/s_map.pkl"):
    # do the correlation voxelwise, ISC vs consensus
    print('computing s_map')
    for e, emo in tqdm(enumerate(df_emotions[n_emo:])):
        for i in tqdm(range(isc_wholebrain.shape[1])):
            nan_mask = ~np.isnan(df_consensus[:, e])
            s_map[e, i] = pearsonr(isc_wholebrain.T[i][nan_mask], df_consensus[:, e][nan_mask])

    # save s_map to pickle
    with open(f"{data_path}/s_map.pkl", 'wb') as f:
        pickle.dump(s_map, f)
else:
    with open(f"{data_path}/s_map.pkl", 'rb') as f:
        s_map = pickle.load(f)

mask_img = np.load(f"{data_path}/mask_img.npy")
ref_nii = nib.load(f"{data_path}/ref_nii.nii.gz")


def plot_brain_from_np(ref, mask, data, data_name, num_perm, plot=False):
    """
    Given a reference nifti file, a mask, and an isc map, plot the isc map on the reference brain
    :param ref: reference nifti file
    :param mask: mask image
    :param data: vectorized brain data
    :param data_name: filename to save, in format {data_name}_{num_perm}
    :param num_perm: number of permutations (used in filename)
    :param plot: whether to display a few slices of the brain
    :return:
    """
    from nilearn.plotting import plot_stat_map

    mask_coords = np.where(mask)
    isc_img = np.full(ref.shape, np.nan)
    isc_img[mask_coords] = data
    isc_nii = nib.Nifti1Image(isc_img, ref.affine, ref.header)
    nib.save(isc_nii, f'{data_path}/{data_name}_{num_perm}')
    if plot:
        plot_stat_map(
            isc_nii,
            cmap='RdYlBu_r',
            cut_coords=(-61, -20, 8))

        # Plot slices at coordinates 0, -65, 40
        plot_stat_map(
            isc_nii,
            cmap='RdYlBu_r',
            cut_coords=(0, -65, 40))
        plt.show()

        # Plot slices at coordinates -61, -20, 8
        plot_stat_map(
            isc_nii,
            cmap='RdYlBu_r',
            cut_coords=(-61, -20, 8))

        # Plot slices at coordinates 0, -65, 40
        plot_stat_map(
            isc_nii,
            cmap='RdYlBu_r',
            cut_coords=(0, -65, 40))


# plot the s_map for each emotion
# for e, emo in enumerate(emotions[:n_emo]):
#     plot_brain_from_np(ref_nii, mask_img, s_map[e, :, 0], f's_map_{emo}', n_perm)

# the p-value is the proportion of permuted correlations that are more extreme than the observed correlation
p_map = np.empty(shape=(s_map.shape[:2]))
if not os.path.exists(f"{data_path}/p_map_{n_perm}.pkl"):
    print('computing p_map')
    for e, emo in tqdm(enumerate(emotions[:n_emo])):
        for voxel in range(s_map.shape[1]):
            if s_map[e, voxel, 0] < 0:
                p_map[e, voxel] = np.sum(s_map[e, voxel, 0] > perm_vox[e, :, 0]) / n_perm
            else:
                p_map[e, voxel] = np.sum(s_map[e, voxel, 0] < perm_vox[e, :, 0]) / n_perm
        # save p_map to pickle
    with open(f"{data_path}/p_map_{n_perm}.pkl", 'wb') as f:
        pickle.dump(p_map, f)
else:
    for e, emo in tqdm(enumerate(emotions[:n_emo])):
        with open(f"{data_path}/p_map_{n_perm}.pkl", 'rb') as f:
            p_map = pickle.load(f)

# calculate and save the z_map for each emotion
thresh_s_map = deepcopy(s_map[:, :, 0])
z_map = norm.ppf(1 - p_map / 2)
for e, emo in enumerate(emotions[:n_emo]):
    q = fdr_threshold(z_map[e], 0.01)
    q_p = 2*(1-norm.cdf(q))  # p-threshold corresponding to q

    # threshold the s_map using the fdr q value
    thresh = p_map[e] >= q_p
    thresh_s_map[e, thresh] = 0
    # plot the thresholded s_map for each emotion
    plot_brain_from_np(ref_nii, mask_img, thresh_s_map[e], f'thresh_s_map_{emo}_q_01', n_perm)
