"""
ISC with sliding window to capture variations in relationship between ISC and emotional report
"""

import os, sys
import pickle
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

sys.path.append('/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/scripts')

# from Pairwise_Temporal_ISC_vs_consensus import plot_brain_from_np
from ISC_Helper import get_rois, sliding_isc  # , permute_isc_behav
from isc_standalone import phase_randomize
from scipy.stats import pearsonr

# -------------------------------
# File paths
# -------------------------------
data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data/ToyStoryNuisanceRegressed'
func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))

# remove VR7 and 8 temporarily for testing because they are 295 not 300 TRs
func_fns = [fn for fn in func_fns if 'VR7' not in fn and 'VR8' not in fn]

# -------------------------------
# Parameters
# -------------------------------
subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]
subj_ids.sort()
roi_selected = ['wholebrain', 'visualcortex', 'auditory', 'vmPFC', 'ACC', 'PCC', 'insula', 'amygdala', 'NA']
emotions = ['P', 'N', 'M', 'X', 'Cry']  # Positive, Negative, Mixed, Neutral
avg_over_roi = True
spatial = False
pairwise = False
window_size = 30
step_size = 5
n_trs = 300
n_windows = int((n_trs - window_size) / step_size) + 1

# -------------------------------
# Compute and save ISC
# -------------------------------
avg_over_roi_name = "avg" if avg_over_roi else "voxelwise"
spatial_name = "spatial" if spatial else "temporal"
pairwise_name = "pairwise" if pairwise else "group"
task = 'Toy_Story'
roi_mask_path = '/Volumes/BCI/Ambivalent_Affect/rois'
all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii*'))
all_roi_masker = get_rois(all_roi_fpaths)
data_path = '/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/toystory'
figure_path = '/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/figures/toystory'
rating_path = f'/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/coded_df_{task}.nc'
isc_path = f"{data_path}/isc_sliding_{pairwise_name}_n{len(subj_ids)}_{avg_over_roi_name}_roi{len(roi_selected)}_" \
           f"window{window_size}_step{step_size}.pkl"
label_dir = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Toy_Story_Labelled'
task = 'Toy_Story'

# save each 3D image as a nii.gz file, one for each of the 5 emotions
if not os.path.exists(isc_path):
    iscs_roi_selected = sliding_isc(roi_selected, all_roi_masker, func_fns, avg_over_roi=avg_over_roi,
                                    data_path=data_path, spatial=spatial, pairwise=pairwise, n_trs=n_trs,
                                    window_size=window_size, step_size=step_size)
    with open(isc_path, 'wb') as f:
        pickle.dump(iscs_roi_selected, f)
else:
    with open(isc_path, 'rb') as f:
        iscs_roi_selected = pickle.load(f)

# generate sliding window and within each window, compute the correlation between ISC and emotional report
# df = xr.open_dataset(rating_path)
# common_sub = [x for x in subj_ids if x in df.subj_id]
# df = df.sel(subj_id=common_sub) common_sub # subset df with only the subjects we have ISC for
# # df = pd.read_excel('/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Label_Aggregate.xlsx', skiprows=[0])
#
# # Compute the percentage of subjects in each state at each time point
# P_counts = df[df == 'P'].count(axis=1)
# N_counts = df[df == 'N'].count(axis=1)
# M_counts = df[df == 'M'].count(axis=1)
# X_counts = df[df == 'X'].count(axis=1)
# Cry_counts = df[df == 'Cry'].count(axis=1)
#
# n_windows = int((df.shape[0] - window_size) / step_size) + 1
#
# slide_behav = []
# for i in range(n_windows):
#     slide_behav.append([P_counts[i * step_size:i * step_size + n_windows].mean(),
#                         N_counts[i * step_size:i * step_size + n_windows].mean(),
#                         M_counts[i * step_size:i * step_size + n_windows].mean(),
#                         X_counts[i * step_size:i * step_size + n_windows].mean(),
#                         Cry_counts[i * step_size:i * step_size + n_windows].mean()])
# slide_behav = np.array(slide_behav)

slide_behav = np.load(f'{label_dir}/slide_behav_{task}.npy')

# isc_vmpfc = iscs_roi_selected['vmPFC'].mean(axis=1)
# scaled_isc = isc_vmpfc / np.max(isc_vmpfc)
# slide_behav_scaled = slide_behav / np.max(slide_behav)
#
# plt.plot(slide_behav_scaled)
# plt.plot(scaled_isc)
# plt.legend(emotions)
# plt.show()
#
# mask_img = np.load(f"{data_path}/../mask_img.npy")
# ref_nii = nib.load(f"{data_path}/../ref_nii.nii.gz")
# mask_coords = np.where(mask_img)
# isc_img = np.full(ref_nii.shape, np.nan)

# # save each sliding window as a Nifti file
# for i in range(n_windows):
#     if not os.path.exists(f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}'):
#         isc_img[mask_coords] = iscs_roi_selected['wholebrain'][i]
#         isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
#         nib.save(isc_nii, f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}')
#     else:
#         print("Loading sliding ISC from file...")
#         nib.load(f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}')

# # correlate the ISC with the emotional report at each sliding window
# corr_path = f'{data_path}/size{window_size}_step{step_size}.pkl'
# if not os.path.exists(corr_path):
#     corrs = np.empty((len(emotions), len(roi_selected)))
#     for e in emotions:
#         for roi in roi_selected:
#             corr = pearsonr(slide_behav[:, emotions.index(e)], iscs_roi_selected[roi].mean(axis=1))
#             corrs[emotions.index(e), roi_selected.index(roi)] = corr[0]
#     with open(corr_path, 'wb') as f:
#         pickle.dump(corrs, f)
# else:
#     with open(corr_path, 'rb') as f:
#         corrs = pickle.load(f)
#     print("Correlations loaded from file.")

# n_perm = 10000
#
# # perform permutation testing per ROI
# perm = np.empty(shape=(slide_behav.shape[1], len(roi_selected), n_perm, 2))  # emotions, rois, n_perm, r/p values
# perm_path = f'{data_path}/sliding_isc/permutations/size{window_size}_step{step_size}_{n_perm}'
# if not os.path.exists(perm_path):  # only compute if file DNE
#     rng = np.random.default_rng()  # for rng.permutation
#     for e in tqdm(range(slide_behav.shape[1]), position=0):  # number of emotions
#         for r, roi in (rbar := tqdm(enumerate(roi_selected), leave=False)):
#             rbar.set_description(f"Roi {r}")
#             for i in range(n_perm):
#                 perm[e, r, i] = pearsonr(iscs_roi_selected[roi].mean(axis=1),
#                                          rng.permutation(slide_behav[:, e]))
#     # save perm to pickle
#     with open(perm_path, 'wb') as f:
#         pickle.dump(perm, f)
# else:
#     with open(perm_path, 'rb') as f:
#         perm = pickle.load(f)
#     print("Permutation results loaded from file.")
#
# p_val = np.empty(shape=(len(emotions), len(roi_selected)))
# # using the null distribution, get the p-values per roi and emotion
# for e, _ in enumerate(emotions):
#     for r, _ in enumerate(roi_selected):
#         if corrs[e, r] > 0:
#             p_val[e, r] = np.sum(perm[e, r, :, 0] >= corrs[e, r]) / n_perm
#         else:
#             p_val[e, r] = np.sum(perm[e, r, :, 0] <= corrs[e, r]) / n_perm
#
# # to choose the threshold, use full Bonferroni correction for number of rois
# # at a 0.05 level, 0.05 / number of rois is the threshold and the p-values that are below this are significant
# thresh = 0.05 / len(roi_selected)
#
# sig = p_val[:]
# sig[sig > thresh] = np.nan
#
# # add labels to rows and columns
# sig = pd.DataFrame(sig, index=emotions, columns=roi_selected)
#
# p_val = np.empty(shape=(len(emotions), len(roi_selected)))
# for e, _ in enumerate(emotions):
#     for r, _ in enumerate(roi_selected):
#         if corrs[e, r] > 0:
#             p_val[e, r] = np.sum(perm[e, r, :, 0] >= corrs[e, r]) / n_perm
#         else:
#             p_val[e, r] = np.sum(perm[e, r, :, 0] <= corrs[e, r]) / n_perm
#
# # to choose the threshold, use full Bonferroni correction for number of rois
# # at a 0.05 level, 0.05 / number of rois is the threshold and the p-values that are below this are significant
# thresh = 0.05 / len(roi_selected)
#
# sig = p_val[:]
# sig[sig > thresh] = np.nan
#
# # add labels to rows and columns
# sig = pd.DataFrame(sig, index=emotions, columns=roi_selected)
#
# # add a star to significant values on the plot
# plt.figure()
# for i, e in enumerate(emotions):
#     plt.bar(np.arange(len(roi_selected)) + i * 0.2, corrs[i, :], width=0.2, label=e, color=['red', 'blue', 'purple', 'dimgray'][i])
#     for j, r in enumerate(roi_selected):
#         if not np.isnan(sig.iloc[i, j]):
#             if corrs[i, j] > 0:
#                 plt.text(j + i * 0.2, corrs[i, j] + 0.01, '*', color='k', fontsize=12, ha='center', va='center',
#                          fontweight='bold')
#             else:
#                 plt.text(j + i * 0.2, corrs[i, j] - 0.04, '*', color='k', fontsize=12, ha='center', va='center',
#                          fontweight='bold')
# plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# plt.ylabel("Pearson's r")
# plt.legend(['Positive', 'Negative', 'Mixed', 'Neutral'])
# plt.title(f"Correlation between sliding window ISC and feeling consensus")
# plt.show()
#
# # add a star to significant values on the plot
# plt.figure()
# for i, e in enumerate(emotions[1:]):
#     plt.bar(np.arange(len(roi_selected)) + i * 0.2, corrs[i+1, :], width=0.2, label=e)
#     for j, r in enumerate(roi_selected):
#         if not np.isnan(sig.iloc[i+1, j]):
#             if corrs[i+1, j] > 0:
#                 plt.text(j + i * 0.2, corrs[i+1, j] + 0.01, '*', color='k', fontsize=12, ha='center', va='center',
#                          fontweight='bold')
#             else:
#                 plt.text(j + i * 0.2, corrs[i+1, j] - 0.04, '*', color='k', fontsize=12, ha='center', va='center',
#                          fontweight='bold')
# plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# plt.ylabel("Pearson's r")
# plt.legend(['Univalent', 'Mixed', 'Neutral'])
# plt.title(f"Correlation between sliding window ISC and emotional consensus")
# plt.savefig(f"{figure_path}/sliding_size{window_size}_step{step_size}")
# plt.show()
#
# # plot emotion data by itself
# plt.figure()
# for i, e in enumerate(emotions):
#     plt.plot(slide_behav[:, i], label=e)
# plt.legend(emotions)
# plt.title("Emotional consensus over time")
# plt.show()

# create correlation matrix for all emotions, showing only one half and listing the value of the correlation in the square
all_emo = np.load(f'{label_dir}/counts_{task}.npy')
mat = np.corrcoef(all_emo.T)
mat = np.tril(mat)
plt.figure()
plt.imshow(mat, cmap='coolwarm')
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if i >= j:
            plt.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center')
plt.colorbar()
plt.xticks(range(len(emotions)), ['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
plt.yticks(range(len(emotions)), ['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
plt.title("Correlation matrix of emotion consensus")
# plt.savefig(f"{figure_path}/consensus_corr_matrix")
plt.show()

# import statsmodels.api as sm
#
# betas = np.empty(shape=(len(roi_selected), len(emotions)))
# pvalues = np.empty(shape=(len(roi_selected), len(emotions)))
# for r, roi in enumerate(roi_selected):
#     model = sm.OLS(iscs_roi_selected[roi].mean(axis=1), slide_behav[:, :4])
#     results = model.fit()
#     betas[r] = results.params
#     pvalues[r] = results.pvalues
# # print roi, emotion,  coef and pvalue if pvalue < 0.05
# for r, roi in enumerate(roi_selected):
#     for e, emo in enumerate(emotions):
#         if pvalues[r, e] < 0.05:
#             print(f"{roi}, {emo}: beta = {betas[r, e]:.3f}, p = {pvalues[r, e]:.3f}")
#
#
# model = sm.OLS(iscs_roi_selected[roi].mean(axis=1), slide_behav[:, :4])
# results = model.fit_regularized(alpha=1.0, L1_wt=0.0)
# results.summary()

# add labels for roi
# plt.figure()
# for i, e in enumerate(emotions):
#     plt.bar(np.arange(len(roi_selected)) + i * 0.2, betas[:, i], width=0.2, label=e)
# plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# plt.ylabel("Beta")
# plt.legend(emotions)
# plt.title(f"Betas for each ROI")
# plt.show()

from sklearn.linear_model import RidgeCV
coefs = np.empty(shape=(len(roi_selected), len(emotions)))
means = []
stds = []
r2s = []
for r, roi in enumerate(roi_selected):
    alpha_range = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alpha_range, store_cv_results=True)
    results = model.fit(slide_behav, iscs_roi_selected[roi].flatten())
    mses = results.cv_results_
    coefs[r] = results.coef_
    mean_mses = np.mean(mses, axis=0)
    std_mses = np.std(mses, axis=0)

    means.append(mean_mses[np.argmin(mean_mses)])
    stds.append(std_mses[np.argmin(std_mses)])
    r2s.append(model.score(slide_behav, iscs_roi_selected[roi].flatten()))

[print(f"emotion consensus explains {r2s[r]:.2f} of variance in {roi}") for r, roi in enumerate(roi_selected)]

plt.figure()
for i, e in enumerate(emotions):
    plt.bar(np.arange(len(roi_selected)) + i * 0.2, coefs[:, i], width=0.2, label=e, color=['red', 'blue', 'purple', 'dimgray', 'brown'][i])
plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
plt.ylabel("Beta value")
plt.legend(['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
plt.title("Emotion consensus vs. ISC")
plt.tight_layout()
plt.show()

phase_randomize()
# coefs = []
# all_results = []
# for r, roi in enumerate(roi_selected):
#     model = RidgeCV(alphas=alpha_range, store_cv_results=True)
#     results = model.fit(slide_behav, iscs_roi_selected[roi])
#     all_results.append(results)
#     coefs.append(results.coef_)
