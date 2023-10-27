"""
This script computes pairwise temporal ISC for a set of ROIs and compares it to pairwise behavioral ratings.
"""
import itertools
import math
import os
import pickle
from glob import glob
from os.path import join
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.glm import fdr_threshold

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
rating_path = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Label_Aggregate.xlsx'

# -------------------------------
# Parameters
# -------------------------------
subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]
roi_selected = ['auditory', 'ACC', 'vmPFC', 'insula', 'visualcortex', 'amygdala', 'wholebrain']  # ['auditory',
# 'visualcortex', 'ACC', 'vmPFC', 'vPCUN', 'aINS_L', 'aANG_L', 'pANG_L', 'Insular_R', 'dPCUN', 'aANG_R', 'aCUN',
# 'pANG_R', 'PMC_L', 'dPCC', 'insula', 'amygdala', 'wholebrain']
emotions = ['P', 'N', 'M', 'X']
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


# Average across ROI with Fisher's z transform
# avg_z = pd.DataFrame(np.array([np.tanh(np.mean(np.arctanh(iscs_roi_selected[roi]), axis=1)) for roi in
#                                roi_selected]).T, columns=roi_selected, index=subj_mapping)

# -------------------------------
# Behavioral ratings
# -------------------------------
df = pd.read_excel(rating_path, header=0, skiprows=[0])  # needs openpyxl
df = df.loc[:, df.columns.isin(subj_ids)]  # subset df with only the subjects we have ISC for
n_pairs = math.comb(df.shape[1], 2)


df_agree_by_emotion = np.empty(shape=(n_pairs, df.shape[0], len(emotions)))
for idx, (col_a, col_b) in enumerate(itertools.combinations(df.columns, 2)):
    for t in range(df.shape[0]):
        for e in range(len(emotions)):
            df_agree_by_emotion[idx, t, e] = int(df[col_a][t] == df[col_b][t] == emotions[e])  # at each time point

window_size = 10
rolling_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'),
                                   axis=1, arr=df_agree_by_emotion)
df_consensus = np.mean(rolling_mean, axis=1)

# corrs = np.empty(shape=(iscs_roi_selected['insula'].shape[1], len(emotions), 2))
# for voxel in range(iscs_roi_selected['insula'].shape[1]):
#     for emotion in range(len(emotions)):
#         for i in range(2):
#             corrs[voxel, emotion, i] = pearsonr(iscs_roi_selected['insula'][:, voxel], df_consensus[:, emotion])[i]

# now make corrs_roi which is the same thing but for all ROIs
# corrs_roi = dict()
# for roi in roi_selected:
#     corrs_roi[roi] = np.empty(shape=(iscs_roi_selected[roi].shape[1], len(emotions), 2))
#     for voxel in range(iscs_roi_selected[roi].shape[1]):
#         for emotion in range(len(emotions)):
#             for i in range(2):
#                 corrs_roi[roi][voxel, emotion, i] = pearsonr(iscs_roi_selected[roi][:, voxel],
#                                                              df_consensus[:, emotion])[i]

# create corrs_roi, a dictionary of keys = roi, values = pd dataframe of shape (n_emotions, 2)
# corrs_roi = dict()
# for roi in roi_selected:
#     corrs_roi[roi] = pd.DataFrame(np.empty(shape=(len(emotions), 2)),
#                                   index=emotions, columns=['r', 'p'])
#     for e, emotion in enumerate(emotions):
#         corrs_roi[roi].loc[[emotion]] = pearsonr(avg_z[roi], df_consensus[:, e])
#
#     # print all significant correlations
# for roi in roi_selected:
#     for emotion in range(len(emotions)):
#         if corrs_roi[roi].loc[emotions[emotion]]['p'] < 0.05:
#             print(f"{roi} and {emotions[emotion]}: {corrs_roi[roi].loc[emotions[emotion]]['r']:.3f}, "
#                   f"{corrs_roi[roi].loc[emotions[emotion]]['p']:.3f}")


n_perm = 100
alpha = int(n_perm * 0.05)
isc_wholebrain = iscs_roi_selected['wholebrain']
n_emo = 3

# loop through emotions pos, neg, mix, (skip neutral)
perm = np.empty(shape=(n_emo, n_perm, isc_wholebrain.shape[1], 2))  # 3 emotions, n_perm, n_voxels, r and p
perm_path = f"{data_path}/perm_{n_perm}.pkl"
if not os.path.exists(perm_path):  # only compute if file DNE
    rng = np.random.default_rng()
    for e, emo in tqdm(enumerate(emotions[:n_emo])):
        for i in tqdm(range(n_perm)):  # number of permutations to loop over
            for j in range(isc_wholebrain.shape[1]):  # number of voxels
                perm[e, i, j] = pearsonr(isc_wholebrain.T[j], rng.permutation(df_consensus[:, e]))

    # save perm to pickle
    with open(perm_path, 'wb') as f:
        pickle.dump(perm, f)
else:
    with open(perm_path, 'rb') as f:
        perm = pickle.load(f)

s_map = np.empty(shape=(n_emo, iscs_roi_selected['wholebrain'].shape[1], 2))
if not os.path.exists(f"{data_path}/s_map.pkl"):
    # do the correlation voxelwise, ISC vs consensus
    print('computing s_map')
    for e, emo in tqdm(enumerate(emotions[:n_emo])):
        for i in range(isc_wholebrain.shape[1]):
            s_map[e, i] = pearsonr(isc_wholebrain.T[i], df_consensus[:, e])

    # save s_map to pickle
    with open(f"{data_path}/s_map.pkl", 'wb') as f:
        pickle.dump(s_map, f)

else:
    with open(f"{data_path}/s_map.pkl", 'rb') as f:
        s_map = pickle.load(f)

assert np.sum(s_map) > 0
assert np.sum(perm) > 0

# view histogram
plt.hist(perm[0, :, 0, 0], bins=100)
plt.title('Histogram of permuted correlations for pos emotion in one voxel')
plt.show()

# get the 95% confidence threshold based on permutation tests
# vox = perm[:, :, 0, 0].deepcopy()
# vox.sort()
# thresh = vox[-alpha]

# now get a 95% threshold for each voxel using np argsort
thresh = np.empty(shape=(n_emo, iscs_roi_selected['wholebrain'].shape[1], ))
for voxel in range(perm.shape[1]):
    thresh = perm[np.argsort(perm[:, :, voxel, 0], axis=0)[-alpha], voxel, 0]

mask_img = np.load(f"{data_path}/mask_img.npy")
ref_nii = nib.load(f"{data_path}/ref_nii.nii.gz")


def plot_brain_from_np(ref, mask, data, data_name):
    """
    Given a reference nifti file, a mask, and an isc map, plot the isc map on the reference brain
    :param ref: reference nifti file
    :param mask: mask image
    :param data: vectorized brain data
    :param data_name: filename to save
    :return:
    """
    from nilearn.plotting import plot_stat_map

    mask_coords = np.where(mask)
    isc_img = np.full(ref.shape, np.nan)
    isc_img[mask_coords] = data
    isc_nii = nib.Nifti1Image(isc_img, ref.affine, ref.header)
    nib.save(isc_nii, f'{data_path}/{data_name}_{n_perm}')
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


plot_brain_from_np(ref_nii, mask_img, s_map[:, 0], 's_map')

# create a p_map which contains the p-value of the s_map using the permutation tests
# p_map = np.empty(shape=(s_map.shape[0]))
# for voxel
