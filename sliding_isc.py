"""
ISC with sliding window to capture variations in relationship between ISC and emotional report
"""

import os
import pickle
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

from ISC_Helper import get_rois, sliding_isc, permute_isc_behav
from scipy.stats import pearsonr

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
perm_path = data_path + '/sliding_isc/permutations'

# -------------------------------
# Parameters
# -------------------------------
subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]
subj_ids.sort()
roi_selected = ['auditory', 'ACC', 'vmPFC', 'insula', 'visualcortex', 'amygdala', 'wholebrain']
emotions = ['P', 'N', 'M', 'X', 'Cry']
spatial = False
pairwise = False
window_size = 30
step_size = 5

# -------------------------------
# Compute and save ISC
# -------------------------------
all_roi_masker = get_rois(all_roi_fpaths)
spatial_name = "spatial" if spatial else "temporal"
pairwise_name = "pairwise" if pairwise else "group"
isc_path = f"{data_path}/isc_sliding_{pairwise_name}_n{len(subj_ids)}_roi{len(roi_selected)}_" \
           f"window{window_size}_step{step_size}.pkl"

# save each 3D image as a nii.gz file, one for each of the 5 emotions
if not os.path.exists(isc_path):
    iscs_roi_selected = sliding_isc(roi_selected, all_roi_masker, func_fns, data_path=data_path, spatial=spatial,
                                    pairwise=pairwise, n_trs=454, window_size=window_size, step_size=step_size)
    with open(isc_path, 'wb') as f:
        pickle.dump(iscs_roi_selected, f)
else:
    with open(isc_path, 'rb') as f:
        iscs_roi_selected = pickle.load(f)

# generate sliding window and within each window, compute the correlation between ISC and emotional report
# df = xr.open_dataset(rating_path)
# df = df.sel(subj_id=subj_ids)  # subset df with only the subjects we have ISC for
df = pd.read_excel('/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Label_Aggregate.xlsx', skiprows=[0])

# Compute the percentage of subjects in each state at each time point
P_counts = df[df == 'P'].count(axis=1)
N_counts = df[df == 'N'].count(axis=1)
M_counts = df[df == 'M'].count(axis=1)
X_counts = df[df == 'X'].count(axis=1)
Cry_counts = df[df == 'Cry'].count(axis=1)

n_windows = int((df.shape[0] - window_size) / step_size) + 1

slide_behav = []
for i in range(n_windows):
    slide_behav.append([P_counts[i * step_size:i * step_size + n_windows].mean(),
                        N_counts[i * step_size:i * step_size + n_windows].mean(),
                        M_counts[i * step_size:i * step_size + n_windows].mean(),
                        X_counts[i * step_size:i * step_size + n_windows].mean(),
                        Cry_counts[i * step_size:i * step_size + n_windows].mean()])
slide_behav = np.array(slide_behav)

isc_vmpfc = iscs_roi_selected['vmPFC'].mean(axis=1)
scaled_isc = isc_vmpfc / np.max(isc_vmpfc)
slide_behav_scaled = slide_behav / np.max(slide_behav)

plt.plot(slide_behav_scaled)
plt.plot(scaled_isc)
plt.legend(['P', 'N', 'M', 'X', 'Cry'])
plt.show()

mask_img = np.load(f"{data_path}/mask_img.npy")
ref_nii = nib.load(f"{data_path}/ref_nii.nii.gz")
mask_coords = np.where(mask_img)
isc_img = np.full(ref_nii.shape, np.nan)

# save each sliding window as a Nifti file
for i in range(n_windows):
    isc_img[mask_coords] = iscs_roi_selected['wholebrain'][i]
    isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
    nib.save(isc_nii, f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}')

# correlate the ISC with the emotional report at each sliding window
corrs = np.empty((len(emotions), len(roi_selected)))
for e in emotions:
    for roi in roi_selected:
        corr = pearsonr(slide_behav[:, emotions.index(e)], iscs_roi_selected[roi].mean(axis=1))
        corrs[emotions.index(e), roi_selected.index(roi)] = corr[0]

plt.figure()
for i, e in enumerate(emotions):
    plt.bar(np.arange(len(roi_selected)) + i * 0.2, corrs[i, :], width=0.2, label=e)
plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
plt.legend()
plt.title(f"Correlation between ISC and emotional report")
plt.show()

n_perm = 1000000
# perm = {}
# for roi in roi_selected:
#     perm[roi] = permute_isc_behav(iscs_roi_selected[roi], slide_behav, n_perm, voxel_idx=100,
#                                   perm_path=f'{perm_path}/size{window_size}_step{step_size}_{n_perm}_{roi}')

perm_wholebrain = permute_isc_behav(iscs_roi_selected['wholebrain'], slide_behav, n_perm, voxel_idx=1000,
                                    perm_path=f'{perm_path}/size{window_size}_step{step_size}_{n_perm}_wholebrain')

