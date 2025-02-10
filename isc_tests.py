#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
ISC with sliding window to capture variations in relationship between ISC and emotional report
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import cProfile
import time
from glob import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from isc_standalone import p_from_null, isc, load_boolean_mask
from ISC_Helper import get_rois, _compute_phaseshift_sliding_isc, load_roi_data
import nibabel as nib
from nilearn import plotting

# -------------------------------
# Parameters
# -------------------------------
task = 'onesmallstep'
roi_selected = ['visualcortex', 'auditory', 'vmPFC', 'ACC', 'PCC', 'insula', 'amygdala', 'NA']
# roi_selected = ['PCC', 'ACC']
emotions = ['P', 'N', 'M', 'X', 'Cry']  # Positive, Negative, Mixed, Neutral
avg_over_roi = True
spatial = False
pairwise = False
random_state = None
window_size = 30
step_size = 5
if task == 'toystory':
    n_trs = 274
    n_shifts = 10000
elif task == 'onesmallstep':
    n_trs = 454
    n_shifts = 10000
else:
    raise Exception('task not defined')
n_windows = int((n_trs - window_size) / step_size) + 1
batch_size = 16

smooth = 'smooth'
avg_over_roi_name = "avg" if avg_over_roi else "voxelwise"
spatial_name = "spatial" if spatial else "temporal"
pairwise_name = "pairwise" if pairwise else "group"

# -------------------------------
# File paths
# -------------------------------
if task == 'toystory':
    data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data/ToyStoryNuisanceRegressed'
elif task == 'onesmallstep':
    data_dir_func = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/ISC_Data_cut/NuisanceRegressed'
else:
    raise ValueError('Invalid task')
func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))

if task == 'toystory':
    # remove VR7 and 8 temporarily for testing because they are 295 not 300 TRs
    func_fns = [fn for fn in func_fns if 'VR7' not in fn and 'VR8' not in fn]
    label_dir = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Toy_Story_Labelled'
elif task == 'onesmallstep':
    label_dir = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/OSS_Labelled'

subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]  # assume BIDS format
subj_ids.sort()

roi_mask_path = '/Volumes/BCI/Ambivalent_Affect/rois'
all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii*'))
all_roi_masker = get_rois(all_roi_fpaths)
data_path = f'/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/{task}'
figure_path = f'/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/figures/{task}'
isc_path = f"{data_path}/isc_sliding_{pairwise_name}_n{len(subj_ids)}_{avg_over_roi_name}_roi{len(roi_selected)}_" \
           f"window{window_size}_step{step_size}.pkl"
sliding_perm_path = f"{data_path}/sliding_isc/permutations/phaseshift_size{window_size}_step{step_size}"
save_path = f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois"
print(save_path)


# In[2]:


parc = nib.load('/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/onesmallstep/schaefer_2018/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii.gz')
print(parc.get_fdata().flatten().shape)
mask = np.load('/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/onesmallstep/mask_img.npy')
assert np.all(parc.shape == mask.shape)
masked_parc = parc.get_fdata().flatten()[mask.flatten()]


# In[62]:


# isc_data_path = f"{data_path}/func_data_parcellated"

# if not os.path.exists(isc_data_path):
#     from nilearn.datasets import fetch_atlas_schaefer_2018
#     from nilearn.maskers import NiftiLabelsMasker

#     all_parcels = []
#     n_parcels = 1000
#     atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=17, resolution_mm=2, data_dir=data_path)
#     labels = [x.decode('UTF-8') for x in atlas.labels]  # https://stackoverflow.com/questions/23618218/numpy-bytes-to-plain-string

#     # Initialize labels masker with atlas parcels
#     masker = NiftiLabelsMasker(atlas.maps, labels=labels)

#     for file in func_fns[:2]:
#         # Fit masker to extract mean time series for parcels
#         all_parcels.append(masker.fit_transform(file))

#     all_parcels = np.array(all_parcels).transpose(1, 2, 0)
#     np.save(isc_data_path, all_parcels)
# else:
#     # load in parcellated data from file
#     all_parcels = np.load(isc_data_path)


# In[3]:


wholebrain_paths = glob(join(data_path, "bold_roi", "wholebrain_*"))

all_parcel_data = []
for p in wholebrain_paths:
    z = np.load(p)
    n_parcels = 1000
    parcel_ts = np.zeros((z.shape[0], n_parcels))
    for parcel_id in range(1, n_parcels + 1):
        parcel_voxels = np.where(masked_parc == parcel_id)[0]
        if parcel_voxels.size > 0:
            parcel_ts[:, parcel_id - 1] = np.mean(z[:, parcel_voxels], axis=1)

    all_parcel_data.append(parcel_ts)


# In[10]:


all_parcel_data.shape


# In[11]:


isc_data_path = f"{data_path}/bold_roi/all_func_data_parcellated"
all_parcel_data = np.array(all_parcel_data).transpose(1,2,0)
np.save(isc_data_path, all_parcel_data)


# In[ ]:


# compute standard temporal ISC parcelwise using a leave-one-out approach
parcel_isc = isc(all_parcel_data)
np.save(f"{isc_data_path}_isc", parcel_isc)


# In[13]:


parcel_isc.shape


# In[14]:


mask_name = f"{roi_mask_path}/wholebrain.nii.gz"

# code from https://brainiak.org/tutorials/10-isc/
# Load the brain mask
brain_mask = load_boolean_mask(mask_name)

# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)

brain_nii = nib.load(mask_name)


# In[26]:


coords[2].shape


# In[27]:


parcel_isc.shape


# In[ ]:


# Plot the time series for an example parcel
from nilearn.plotting import plot_stat_map

example_parcel = 195
func_parcel = parcel_isc[:, example_parcel]

fig, ax = plt.subplots(figsize=(8, 2))
ax.plot(func_parcel)
ax.set(xlabel='TRs', ylabel='activity', xlim=(0, len(func_parcel)))
sns.despine()

# Plot parcel on MNI atlas
parcels_label = np.zeros(func_parcels.shape[1])
parcels_label[example_parcel] = 1

# Invert masker transform to project onto brain
parcel_img = masker.inverse_transform(parcels_label)
plot_stat_map(parcel_img, cmap='Blues');


# In[15]:


# # Make the ISC output a volume
# isc_vol = np.zeros(brain_nii.shape)
# # Map the ISC data for the first participant into brain space
# isc_vol[coords] = parcel_isc[0]
# # make a nii image of the isc map 
# isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)

# # Save the ISC data as a volume
# isc_map_path = f"{data_path}/ISC_{parcel_isc.shape[0]}.nii.gz"
# nib.save(isc_nifti, isc_map_path)
# # Plot the data as a statmap
# threshold = .2

# from nilearn import plotting
# f, ax = plt.subplots(1,1, figsize = (12, 5))
# plotting.plot_stat_map(
#     isc_nifti, 
#     threshold=threshold, 
#     axes=ax
# )
# # ax.set_title('ISC map for subject {}, task = {}' .format(subj_id,task_name)) 

