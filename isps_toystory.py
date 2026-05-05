"""
ISPS for onesmallstep — bandpass filter, parcellate, compute empirical phase synchrony, save to disk.
Also computes a phase-randomization null distribution (same approach as sliding_isc_toystory.py).
"""

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from os.path import join

import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from ISC_Helper import get_rois, load_roi_data, load_schaeffer1000, _compute_isps, _compute_phaseshift_isps
from isc_standalone import p_from_null

# -------------------------------
# Parameters
# -------------------------------
task = 'onesmallstep'
roi_selected = ['wholebrain']
tr = 1.0            # seconds
low_hz, high_hz = 0.04, 0.07   # Glerean et al. 2012
n_trs = 454
n_parcels = 1000
n_shifts = 1024
batch_size = 16

# -------------------------------
# File paths
# -------------------------------
base_dir = '/usr/people/ri4541/juke/isc'
scripts_path = f'{base_dir}/isc_scripts'
data_dir_func = f'{base_dir}/outputs/{task}/data/nuisance_regressed_cut'
func_fns = (glob(join(data_dir_func, 'P?.nii.gz'))  + glob(join(data_dir_func, 'N?.nii.gz'))  +
            glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) +
            glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz')))
n_subj = len(func_fns)
print(f'Found {n_subj} functional files.')

roi_mask_path = f'{scripts_path}/rois'
all_roi_masker = get_rois(glob(os.path.join(roi_mask_path, '*.nii*')))
data_path = f'{base_dir}/outputs/{task}/data'
mask_path = f'{data_path}/mask_img.npy'
parc_path = f'{scripts_path}/schaefer_2018/Schaefer2018_{n_parcels}Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
isps_path = f'{data_path}/isps_{n_subj}subj_{n_parcels}parcels_{len(roi_selected)}rois.pkl'
isps_perm_path = (f'{data_path}/sliding_isc/permutations/'
                  f'phaseshift_size30_step5_{n_parcels}parcelsparcellated_{n_shifts}perms_{len(roi_selected)}rois_isps_x')

_, masked_parc = load_schaeffer1000(parc_path, mask_path)


def bandpass(data, low, high, tr, order=4):
    """Butterworth bandpass along axis=0. data: (n_trs, n_voxels, n_subjects)."""
    nyq = 0.5 / tr
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def parcellate(data, n_parcels, masked_parc):
    """Average voxels within each parcel. data: (n_trs, n_voxels, n_subjects) → (n_trs, n_parcels, n_subjects)."""
    parcel_ts = np.zeros((data.shape[0], n_parcels, data.shape[2]))
    for parcel_id in range(1, n_parcels + 1):
        voxels = np.where(masked_parc == parcel_id)[0]
        if voxels.size > 0:
            parcel_ts[:, parcel_id - 1, :] = data[:, voxels, :].mean(axis=1)
    return parcel_ts


if __name__ == '__main__':
    # --- Step 1: observed ISPS ---
    if not os.path.exists(isps_path):
        isps_roi = {}
        for j, roi_name in enumerate(roi_selected):
            print(j, roi_name)
            bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path)
            bold_filtered = bandpass(bold_roi, low_hz, high_hz, tr)
            bold_parcellated = parcellate(bold_filtered, n_parcels, masked_parc)
            print(f'  parcellated shape: {bold_parcellated.shape}')
            isps_roi[roi_name] = _compute_isps(bold_parcellated, assume_narrowband=True)
            print(f'  ISPS shape: {isps_roi[roi_name].shape}')
        with open(isps_path, 'wb') as f:
            pickle.dump(isps_roi, f)
        print('Saved ISPS to', isps_path)
    else:
        print('Already exists:', isps_path)

    # --- Step 2: phase-shift null distribution ---
    os.makedirs(os.path.dirname(isps_perm_path), exist_ok=True)
    if not os.path.exists(isps_perm_path):
        print(f'\nComputing {n_shifts} phase-shift permutations...')
        n_shifts_per_batch = n_shifts // batch_size

        isps_perm_batches = {roi: [] for roi in roi_selected}
        for j, roi_name in enumerate(roi_selected):
            print(f'ROI: {roi_name}')
            bold_roi = load_roi_data(roi_name, all_roi_masker, func_fns, data_path)
            bold_filtered = bandpass(bold_roi, low_hz, high_hz, tr)
            bold_parcellated = parcellate(bold_filtered, n_parcels, masked_parc)

            start = time.time()
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        _compute_phaseshift_isps,
                        bold_parcellated,
                        n_shifts_per_batch,
                        None,  # random_state — each thread gets its own
                    ): b
                    for b in range(batch_size)
                }
                for future in tqdm(as_completed(futures), total=batch_size, desc='batches'):
                    try:
                        isps_perm_batches[roi_name].append(future.result())
                    except Exception as e:
                        print(f'Batch failed: {e}')
            print(f'  Permutations done in {time.time() - start:.1f}s')

        # Combine batches into x[roi] = [observed, p, distribution]
        # matching the sliding ISC save format
        x = {}
        for roi_name in roi_selected:
            batches = isps_perm_batches[roi_name]
            observed = batches[0][0]  # (n_trs, n_parcels) — same across batches
            distribution = np.concatenate([b[1] for b in batches], axis=0)  # (n_shifts, n_trs, n_parcels)
            p = p_from_null(observed, distribution, side='two-sided', exact=False, axis=0)
            x[roi_name] = [observed, p, distribution]
            print(f'{roi_name}: observed {observed.shape}, distribution {distribution.shape}')

        with open(isps_perm_path, 'wb') as f:
            pickle.dump(x, f)
        print('Saved permutations to', isps_perm_path)
    else:
        print('Already exists:', isps_perm_path)
