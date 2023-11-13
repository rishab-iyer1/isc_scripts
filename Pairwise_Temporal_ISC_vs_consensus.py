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
# df_agree_by_emotion = np.empty(shape=(n_pairs, df.dims['TR'], len(emotions)))
# for idx, (col_a, col_b) in enumerate(itertools.combinations(df.columns, 2)):
#     for t in range(df.shape[0]):
#         for e in range(len(emotions)):
#             df_agree_by_emotion[idx, t, e] = int(df[col_a][t] == df[col_b][t] == emotions[e])  # at each time point
df_emotions = ['P', 'N', 'M', 'X', 'Cry', 'P_smooth', 'N_smooth', 'M_smooth', 'X_smooth', 'Cry_smooth']
df_consensus = np.empty(shape=(len(subj_mapping), 5))
for i, (s1, s2) in enumerate(subj_mapping):
    for j, emo in enumerate(df_emotions[5:]):
        df_consensus[i, j] = pearsonr(df.sel(subj_id=s1, emotion=emo).to_array()[0, :],
                                      df.sel(subj_id=s2, emotion=emo).to_array()[0, :])[0]
# list the number of nans in each column
print(np.sum(np.isnan(df_consensus), axis=0))
n_perm = 100000
alpha = int(n_perm * 0.05)
isc_wholebrain = iscs_roi_selected['wholebrain']

# loop through emotions pos, neg, mix, (skip neutral)
# perm = np.empty(shape=(n_emo, n_perm, isc_wholebrain.shape[1], 2))  # 3 emotions, n_perm, n_voxels, r and p
# perm_path = f"{data_path}/perm_{n_perm}.pkl"
# if not os.path.exists(perm_path):  # only compute if file DNE
#     rng = np.random.default_rng()
#     for e, emo in tqdm(enumerate(emotions[:n_emo])):
#         for i in tqdm(range(n_perm)):  # number of permutations to loop over
#             for j in range(isc_wholebrain.shape[1]):  # number of voxels
#                 perm[e, i, j] = pearsonr(isc_wholebrain.T[j], rng.permutation(df_consensus[:, e]))
#     # save perm to pickle
#     with open(perm_path, 'wb') as f:
#         pickle.dump(perm, f)
# else:
#     with open(perm_path, 'rb') as f:
#         perm = pickle.load(f)

# do permutations on just one voxel
# run permutation testing for the brain to consensus correlation to test null hypothesis that there is no correlation
# between one voxel and correlation of behavioral ratings
vox_idx = 1000  # pick just one voxel to do permutations on
perm_vox = np.empty(shape=(df_consensus.shape[1], n_perm, 2))  # number of emotions, n_perm, r and p
perm_vox_path = f"{data_path}/perm_vox_{n_perm}.pkl"  # path to save
if not os.path.exists(perm_vox_path):  # only compute if file DNE
    rng = np.random.default_rng()  # for rng.permutation
    for e in tqdm(range(df_consensus.shape[1])):  # number of emotions
        for i in tqdm(range(n_perm)):  # number of permutations
            nan_mask = ~np.isnan(df_consensus[:, e])  # mask to ignore nans for any given pair
            perm_vox[e, i] = pearsonr(isc_wholebrain.T[vox_idx][nan_mask],
                                      rng.permutation(df_consensus[:, e][nan_mask]))
    # save perm to pickle
    with open(perm_vox_path, 'wb') as f:
        pickle.dump(perm_vox, f)
else:
    with open(perm_vox_path, 'rb') as f:
        perm_vox = pickle.load(f)

# view histogram
plt.hist(perm_vox[0, :, 0], bins=100)
plt.title('Histogram of permuted correlations for pos emotion in one voxel')
plt.show()

# print critical values for each emotion based on permutation testing
for e in range(5):
    print(f"Critical value for {df_emotions[e+5]} at p=0.001: {np.sort(perm_vox[e, :, 0])[-10]:.3f}")

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

# assert np.sum(s_map, axis=[s_map.dims]) > 0
# assert np.sum(perm_vox) > 0

# get the 95% confidence threshold based on permutation tests
# vox = perm[:, :, 0, 0].deepcopy()
# vox.sort()
# thresh = vox[-alpha]

# now get a 95% threshold for each voxel using np argsort
# thresh = np.empty(shape=(n_emo, iscs_roi_selected['wholebrain'].shape[1], ))
# for voxel in range(perm_vox.shape[1]):
#     thresh = perm_vox[np.argsort(perm_vox[:, :, voxel, 0], axis=0)[-alpha], voxel, 0]

# thresh = perm_vox[np.argsort(perm_vox[:, 0], axis=0)[-alpha], 0]

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
for e, emo in enumerate(emotions[:n_emo]):
    plot_brain_from_np(ref_nii, mask_img, s_map[e, :, 0], f's_map_{emo}', n_perm)

# the p-value is the proportion of permuted correlations that are greater than the observed correlation
p_map = np.empty(shape=(s_map.shape[:2]))
for e, emo in enumerate(emotions[:n_emo]):
    for voxel in range(s_map.shape[1]):
        p_map[e, voxel] = np.sum(s_map[e, voxel, 0] <= np.abs(perm_vox[e, :, 0])) / n_perm

# plot the p_map for each emotion
for e, emo in enumerate(emotions[:n_emo]):
    plot_brain_from_np(ref_nii, mask_img, p_map[e], f'p_map_{emo}', n_perm)

p_map[p_map == 0] += 1e-8  # to avoid log(0)
p_map[p_map == 1] -= 1e-8  # to avoid log(0)

# convert to z map
z_map = norm.ppf(1 - (p_map / 2))

# plot the z_map for each emotion
for e, emo in enumerate(emotions[:n_emo]):
    plot_brain_from_np(ref_nii, mask_img, z_map[e], f'z_map_{emo}', n_perm)

# use nilearn.glm.fdr_threshold to get a thresholded map
thresh = np.empty(shape=z_map.shape[0])
for e, emo in enumerate(emotions[:n_emo]):
    thresh[e] = fdr_threshold(z_map[e], alpha=0.05)

# plot the thresholded s_map for each emotion
thresh_s_map = deepcopy(s_map[:, :, 0])
for e, emo in enumerate(emotions[:n_emo]):
    thresh_s_map[e, z_map[e] < thresh[e]] = 0
    plot_brain_from_np(ref_nii, mask_img, thresh_s_map[e], f'thresh_s_map_{emo}', n_perm)
