"""
This script computes pairwise spatial ISC for a set of ROIs and compares it to pairwise behavioral ratings.
"""
import itertools
import math
import os
import pickle
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
roi_selected = ['auditory', 'ACC', 'vmPFC', 'insula', 'visualcortex', 'amygdala', 'wholebrain']  # ['insula']
emotions = ['P', 'N', 'M', 'X']
spatial = True
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

# -------------------------------
# Behavioral ratings
# -------------------------------
df = pd.read_excel(rating_path, header=0, skiprows=[0])  # needs openpyxl
df = df.loc[:, df.columns.isin(subj_ids)]  # subset df with only the subjects we have ISC for
n_pairs = math.comb(df.shape[1], 2)

df_consensus = dict()
for col_a, col_b in itertools.combinations(df.columns, 2):
    df_consensus['{}_{}'.format(col_a, col_b)] = np.sum(df[col_a] == df[col_b]) / df.shape[0]  # mean consensus

df_agree_over_time = dict()
df_agree_by_emotion = np.empty(shape=(n_pairs, df.shape[0], len(emotions)))
for idx, (col_a, col_b) in enumerate(itertools.combinations(df.columns, 2)):
    for t in range(df.shape[0]):
        for e in range(len(emotions)):
            df_agree_by_emotion[idx, t, e] = int(df[col_a][t] == df[col_b][t] == emotions[e])  # at each time point

# the isc has shape (n_combinations, n_TRs) where each cell is the isc for that combination of subjects and TR in an roi
# ratings has one value for each combination
# question is do people with high isc also have high agreement in ratings?
# for i in range(n_pairs):
#     print(pearsonr(np.array(n_pairs)[i], iscs_roi_selected['insula'][i]))

# corr_isc_consensus = pearsonr(np.abs(iscs_roi_selected['insula']).mean(axis=1), list(df_consensus.values()))
# plt.scatter(list(df_consensus.values()), np.abs(iscs_roi_selected['insula']).mean(axis=1))
# plt.xlabel('Consensus')
# plt.ylabel('Mean ISC')
# plt.title('Mean ISC vs Consensus')
# # linear regression
# m, b = np.polyfit(list(df_consensus.values()), np.abs(iscs_roi_selected['insula']).mean(axis=1), 1)
# plt.plot(list(df_consensus.values()), m * np.array(list(df_consensus.values())) + b, label='r = {}'.format(np.round(corr_isc_consensus[0], 2)))
# # print correlation on the graph
# plt.legend()
# plt.show()

window_size = 10
rolling_mean = np.apply_along_axis(lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'),
                                   axis=1, arr=df_agree_by_emotion)

corrs = np.empty(shape=(n_pairs, len(roi_selected), len(emotions), 2))  # 2 for r and p
for i in range(n_pairs):
    for j in range(len(roi_selected)):
        for k in range(len(emotions)):
            for l in range(2):
                corrs[i, j, k, l] = pearsonr(rolling_mean[i, :, k], iscs_roi_selected[roi_selected[j]][i])[l]

nan_mask = np.isnan(corrs)
corrs[nan_mask] = 0

significant_results = []
# Loop through the ndarray and store significant results
for roi_idx, roi in enumerate(roi_selected):
    for emo_idx, emotion in enumerate(emotions):
        r_statistic = corrs[:, roi_idx, emo_idx, 0]
        p_value = corrs[:, roi_idx, emo_idx, 1]

        # Filter significant correlations with p < 0.05
        significant_mask = p_value < 0.05
        significant_correlations = r_statistic[significant_mask]

        if len(significant_correlations) > 0:
            for r in significant_correlations:
                significant_results.append({'ROI': roi, 'Emotion': emotion, 'r_statistic': r})

# Create a pandas DataFrame from the list of significant results
signif_corrs = pd.DataFrame(significant_results)

# -------------------------------
# Some data exploration
# -------------------------------
# Group by Emotion and ROI, then calculate number of significant correlations, max correlation,
# and average magnitude of correlation
grouped_data = signif_corrs.groupby(['Emotion', 'ROI']).agg(
    num_significant=('r_statistic', 'size'),
    max_correlation=('r_statistic', lambda x: np.max(x)),
    min_correlation=('r_statistic', lambda x: np.min(x)),
    avg_correlation=('r_statistic', lambda x: np.mean(x))
).reset_index()
# Find the emotions with the most significant correlations for each ROI
max_num_significant = grouped_data.groupby('ROI')['num_significant'].transform('max')
most_significant_emotions = grouped_data[grouped_data['num_significant'] == max_num_significant]
print(grouped_data)
print("\nMost significant emotions for each ROI:")
print(most_significant_emotions)

# Group by Emotion and calculate the proportion of negative and positive correlations
proportion_df = signif_corrs.groupby('Emotion')['r_statistic'].apply(lambda x: (x < 0).mean()).reset_index()
proportion_df.rename(columns={'r_statistic': 'Proportion_Negative'}, inplace=True)
proportion_df['Proportion_Positive'] = 1 - proportion_df['Proportion_Negative']
print(proportion_df)

# # plot on 2 subplots
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].hist([corrs[:, 0, 0, 0], corrs[:, 0, 1, 0], corrs[:, 0, 2, 0], corrs[:, 0, 3, 0]], label=emotions)
# axs[0].set_title('correlations')
# axs[1].hist([corrs[:, 0, 0, 1], corrs[:, 0, 1, 1], corrs[:, 0, 2, 1], corrs[:, 0, 3, 1]], label=emotions)
# axs[1].legend()
# axs[1].set_title('p-values')
# plt.show()

# plt.hist([corrs[:, 0, 0, 0], corrs[:, 0, 1, 0], corrs[:, 0, 2, 0], corrs[:, 0, 3, 0]], label=emotions)
# plt.legend()
# plt.title('correlations')
# plt.show()
# plt.hist([corrs[:, 0, 0, 1], corrs[:, 0, 1, 1], corrs[:, 0, 2, 1], corrs[:, 0, 3, 1]], label=emotions)
# plt.legend()
# plt.title('p-values')
# corrs = np.array([[pearsonr(list(df_agree_over_time.values())[i], iscs_roi_selected['insula'][i])[0],
#                    pearsonr(list(df_agree_over_time.values())[i], iscs_roi_selected['insula'][i])[1]]
#                   for i in range(n_pairs)])
