"""
ISC with sliding window to capture variations in relationship between ISC and emotional report
"""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cProfile
import time
from glob import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from isc_standalone import p_from_null
from ISC_Helper import get_rois, _compute_phaseshift_sliding_isc, load_roi_data, parcellate_bold, load_schaeffer1000

# -------------------------------
# Parameters
# -------------------------------
task = 'onesmallstep'
# roi_selected = ['visualcortex', 'auditory', 'vmPFC', 'ACC', 'PCC', 'insula', 'amygdala', 'NA']
roi_selected = ['wholebrain']
# emotions = ['P', 'N', 'M', 'X', 'Cry']  # Positive, Negative, Mixed, Neutral, Cry
emotions = ['P', 'N', 'M']
parcellate = True
subset_oss = False
avg_over_roi = False
spatial = False
pairwise = False
random_state = None
window_size = 30
step_size = 5
if task == 'toystory':
    n_trs = 288
    n_shifts = 12
elif task == 'onesmallstep':
    n_trs = 454
    n_shifts = 1024
else:
    raise Exception('task not defined')
n_windows = int((n_trs - window_size) / step_size) + 1
batch_size = 8

smooth = 'smooth'
avg_over_roi_name = "avg" if avg_over_roi else "voxelwise"
spatial_name = "spatial" if spatial else "temporal"
pairwise_name = "pairwise" if pairwise else "group"

# -------------------------------
# File paths
# -------------------------------
if task == 'toystory':
    data_dir_func = '/jukebox/norman/rsiyer/isc/toystory/nuisance_regressed_cut'
elif task == 'onesmallstep':
    data_dir_func = '/jukebox/norman/rsiyer/isc/outputs/onesmallstep/data/nuisance_regressed_cut'
else:
    raise ValueError('Invalid task')

assert os.path.exists(data_dir_func)

func_fns = glob(join(data_dir_func, 'P?.nii.gz')) + glob(join(data_dir_func, 'N?.nii.gz')) + \
           glob(join(data_dir_func, 'VR?.nii.gz')) + glob(join(data_dir_func, 'P??.nii.gz')) + \
           glob(join(data_dir_func, 'N??.nii.gz')) + glob(join(data_dir_func, 'VR??.nii.gz'))

if task == 'toystory':
    label_dir = '/jukebox/norman/rsiyer/isc/VideoLabelling/Toy_Story_Labelled'
elif task == 'onesmallstep':
    label_dir = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/OSS_Labelled'

subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]  # assume BIDS format
subj_ids.sort()

roi_mask_path = '/jukebox/norman/rsiyer/isc/isc_scripts/rois'
all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii*'))
all_roi_masker = get_rois(all_roi_fpaths)
data_path = f'/jukebox/norman/rsiyer/isc/outputs/{task}/data'
figure_path = f'/jukebox/norman/rsiyer/isc/outputs/{task}/figures'
parc_path = f"/jukebox/norman/rsiyer/isc/isc_scripts/schaefer_2018/Schaefer2018_300Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
mask_path = f"{data_path}/mask_img.npy"
isc_path = f"{data_path}/isc_sliding_{pairwise_name}_n{len(subj_ids)}_{avg_over_roi_name}_roi{len(roi_selected)}_" \
           f"window{window_size}_step{step_size}.pkl"
sliding_perm_path = f"{data_path}/sliding_isc/permutations/phaseshift_size{window_size}_step{step_size}_300parcels"

if parcellate:
    assert avg_over_roi is False
    sliding_perm_path += "parcellated"
    n_parcels = 300
    masked_parc = load_schaeffer1000(parc_path, mask_path)

# -------------------------------
# Compute and save ISC
# -------------------------------

def unpack_and_call(func, kwargs):
    return func(**kwargs)


if __name__ == '__main__':
    # roi_selected = roi_selected[1:]
    # func_fns = func_fns[:3]
    save_path = f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois"
    if not os.path.exists(save_path):
        print("permutation path doesn't exist, computing...")
        from itertools import repeat
        start = time.perf_counter()
        print('roi_selected:', roi_selected)
        with ThreadPoolExecutor() as executor:
            bold_roi = executor.map(load_roi_data, roi_selected, repeat(all_roi_masker), repeat(func_fns), repeat(data_path))  # repeat is used to pass the parameter to each iteration in map(). the 

        end = time.perf_counter()
        print(f"Data loaded in {end-start:.3f} sec")
        
        n_shifts_batch = int(n_shifts/batch_size)
        iscs_roi_selected = dict(zip(roi_selected, [[] for _ in range(len(roi_selected))]))
        with cProfile.Profile() as profile:
            for i, roi in enumerate(bold_roi):
                # print(roi)
                if parcellate:
                    roi = parcellate_bold(roi, n_parcels, masked_parc[0].get_fdata())
                    print(roi.shape)

                if subset_oss:
                    assert task == 'toystory'
                    assert parcellate is True
                    assert roi.shape[1] == 1000
                    # load mask defined from oss
                    mask = pickle.load(open(f"{data_path}/oss_sig_betas_mask.pkl", 'rb'))
                    union_mask = np.unique(np.concatenate(mask))
                    print('subsetting with significant parcels from OSS')
                    print(f"roi shape before subsetting: {roi.shape}")
                    roi = roi[:, union_mask]
                    print(f"roi shape after subsetting: {roi.shape}")

                # Start timing
                start_time = time.time()
                print(f'starting permutations for {roi_selected[i]}')

                with ProcessPoolExecutor() as executor:
                    futures = []
                    for n_batch in (pbar := tqdm([n_shifts_batch]*batch_size)):
                        futures.append(executor.submit(_compute_phaseshift_sliding_isc, data=roi, n_trs=n_trs, window_size=window_size,
                                                                            step_size=step_size,
                                                                            avg_over_roi=avg_over_roi, spatial=spatial,
                                                                            pairwise=pairwise,
                                                                            summary_statistic='median',
                                                                            n_shifts=n_batch,
                                                                            tolerate_nans=True,
                                                                            random_state=random_state))

                    for future in tqdm(as_completed(futures), total=len(futures)):
                        try:
                            iscs_roi_selected[roi_selected[i]].append(future.result())
                            pbar.update(1)
                        except Exception as e:
                            print(f"Task generated an exception: {e}")

                # Print how long the executor took
                print(f"Executor submit and as_completed took {time.time() - start_time:.2f} seconds")
            # results = pstats.Stats(profile)
            # results.sort_stats(pstats.SortKey.TIME)
            # results.print_stats()

        # print(iscs_roi_selected['visualcortex'][0][0].shape, iscs_roi_selected['visualcortex'][0][1].shape, iscs_roi_selected['visualcortex'][0][2].shape)
        # print(iscs_roi_selected['visualcortex'][1][0].shape, iscs_roi_selected['visualcortex'][1][1].shape, iscs_roi_selected['visualcortex'][1][2].shape)
        # print(iscs_roi_selected['auditory'][0][0].shape, iscs_roi_selected['auditory'][0][1].shape, iscs_roi_selected['auditory'][0][2].shape)
        # print(iscs_roi_selected['auditory'][1][0].shape, iscs_roi_selected['auditory'][1][1].shape, iscs_roi_selected['auditory'][1][2].shape)
        with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois", 'wb') as f:
            pickle.dump(iscs_roi_selected, f)
        
        # after parallelizing, the n batches are in n separate elements of phase_slide_isc; recombine them so that we have one nparray of observed, p, distribution for each roi
        x = {roi: [np.empty(shape=(n_windows, 1)), np.empty(shape=(n_windows, 1)), np.empty(shape=(n_shifts, n_windows, 1))] for roi in roi_selected} # init empty dict with appropriate shapes
        for roi in roi_selected:
            # joining the batched distributions
            dist = []
            for i in range(batch_size):  # number of loops = number of batches
                assert np.all(iscs_roi_selected[roi][0][0] == iscs_roi_selected[roi][i][0])  # make sure the "observed" is the same - should never change across batches
                dist.append(iscs_roi_selected[roi][i][2])

            x[roi][0] = iscs_roi_selected[roi][0][0]  # take one of the "observed" ISC matrices since we asserted that they're all the same
            x[roi][2] = np.concatenate(dist)  # concatenate all n_shifts permutations
            x[roi][1] = p_from_null(x[roi][0], x[roi][2], side='two-sided', exact=False, axis=0)  # need to re-calculate p-values after concatenating permutations
        with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois_x", 'wb') as f:
                pickle.dump(x, f)
                print('saved permutations to', f)

    else:
        print(f"File already exists: \n{save_path}")
        with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois_x", 'rb') as f:
            x = pickle.load(f)

        for roi in roi_selected:
            print(f"roi: {roi}")
            print(f"observed shape: {x[roi][0].shape}")
            print(f"p shape: {x[roi][1].shape}")
            print(f"distribution shape: {x[roi][2].shape}")
            print(f"observed mean: {np.mean(x[roi][0])}")
            print(f"p mean: {np.mean(x[roi][1])}")
            print(f"distribution mean: {np.mean(x[roi][2])}")