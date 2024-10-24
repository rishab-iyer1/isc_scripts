"""
ISC with sliding window to capture variations in relationship between ISC and emotional report
"""

import os, sys
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import cProfile
import pstats
import time
from glob import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import RidgeCV

sys.path.append('/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/scripts')
from isc_standalone import p_from_null
from ISC_Helper import get_rois, _compute_phaseshift_sliding_isc, load_roi_data, phaseshift_sliding_isc

# -------------------------------
# Parameters
# -------------------------------
task = 'onesmallstep'
roi_selected = ['wholebrain', 'visualcortex', 'auditory', 'vmPFC', 'ACC', 'PCC', 'insula', 'amygdala', 'NA']
# roi_selected = ['PCC', 'ACC']
emotions = ['P', 'N', 'M', 'X', 'Cry']  # Positive, Negative, Mixed, Neutral
avg_over_roi = True
spatial = False
pairwise = False
random_state = None
window_size = 30
step_size = 5
if task == 'toystory':
    n_trs = 300
elif task == 'onesmallstep':
    n_trs = 484
n_windows = int((n_trs - window_size) / step_size) + 1
n_shifts = 1000
batch_size = 25

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
    pass

subj_ids = [str(subj).split('/')[-1].split('.')[0] for subj in func_fns]  # assume BIDS format
subj_ids.sort()

roi_mask_path = '/Volumes/BCI/Ambivalent_Affect/rois'
all_roi_fpaths = glob(os.path.join(roi_mask_path, '*.nii*'))
all_roi_masker = get_rois(all_roi_fpaths)
data_path = f'/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/{task}'
figure_path = f'/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/figures/{task}'
# rating_path = f'/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/coded_df_{task}.nc'
isc_path = f"{data_path}/isc_sliding_{pairwise_name}_n{len(subj_ids)}_{avg_over_roi_name}_roi{len(roi_selected)}_" \
           f"window{window_size}_step{step_size}.pkl"
sliding_perm_path = f"{data_path}/sliding_isc/permutations/phaseshift_size{window_size}_step{step_size}"


# -------------------------------
# Compute and save ISC
# -------------------------------

# save each 3D image as a nii.gz file, one for each of the 5 emotions
# if not os.path.exists(isc_path):
#     iscs_roi_selected = sliding_isc(roi_selected=roi_selected, all_roi_masker=all_roi_masker, func_fns=func_fns,
#                                     avg_over_roi=avg_over_roi,
#                                     data_path=data_path, spatial=spatial, pairwise=pairwise, n_trs=n_trs,
#                                     window_size=window_size, step_size=step_size)
#     with open(isc_path, 'wb') as f:
#         pickle.dump(iscs_roi_selected, f)
# else:
# with open(isc_path, 'rb') as f:
#     iscs_roi_selected = pickle.load(f)

# for roi in roi_selected:
#     phase_slide_isc = phaseshift_sliding_isc(roi_selected=[f"{roi}"], all_roi_masker=all_roi_masker,
#                                              func_fns=func_fns, n_trs=n_trs, data_path=data_path,
#                                              avg_over_roi=avg_over_roi,
#                                              spatial=spatial, pairwise=pairwise, summary_statistic='median',
#                                              tolerate_nans=True, n_shifts=n_shifts, window_size=window_size,
#                                              step_size=step_size)
#     with open(f"{sliding_perm_path}_{n_shifts}_{roi}", 'wb') as f:
#         pickle.dump(phase_slide_isc, f)
#     del phase_slide_isc


# returns dict of rois and the data for each roi is a tuple containing (observed, p, distribution)

#
# working version
# def main():
#     with cProfile.Profile() as profile:
#         # if not os.path.exists(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois"):
#         with ProcessPoolExecutor() as executor:
#             from time import time
#             from itertools import repeat
#             start = time()
#             phase_slide_isc = list(tqdm(executor.map(phaseshift_sliding_isc, repeat(roi_selected), repeat(all_roi_masker),
#                                                     repeat(func_fns), repeat(n_trs), repeat(data_path),
#                                                     repeat(avg_over_roi),
#                                                     repeat(spatial), repeat(pairwise), repeat('median'),
#                                                     repeat(True), repeat(random_state), [n_shifts/batch_size]*batch_size, repeat(window_size),
#                                                     repeat(step_size)), total=n_shifts))   # n_jobs = None will perform multiprocessing on as many cpus as possible
#             end = time()
#             print(f"{(end-start):.2f} seconds")
#         results = pstats.Stats(profile)
#         results.sort_stats(pstats.SortKey.TIME)
#         results.print_stats()
#         with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois", 'wb') as f:
#             pickle.dump(phase_slide_isc, f)
#     else:
# with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois", 'rb') as f:
#     phase_slide_isc = pickle.load(f)

# trying to parallelize and optimize memory by not loading rois many times

def unpack_and_call(func, kwargs):
    return func(**kwargs)


def main():
    # func_fns = func_fns[:3]
    # roi_selected = roi_selected[:2]
    # bold_roi = []
    # for roi_name in roi_selected:
    #     print(f"Loading {roi_name} data")
    #     bold_roi.append(load_roi_data(roi_name, all_roi_masker, func_fns, data_path))  # shape (n_TRs, n_voxels, n_subjects)

    from itertools import repeat
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        bold_roi = executor.map(load_roi_data, roi_selected, repeat(all_roi_masker), repeat(func_fns), repeat(data_path))  # repeat is used to pass the parameter to each iteration in map(). the 

    end = time.perf_counter()
    print(f"Data loaded in {end-start:.3f} sec")
            
    # err
    from functools import partial
    n_shifts_batch = int(n_shifts/batch_size)
    kwargs = [{"n_shifts":n_shifts_batch} for _ in range(batch_size)]
    iscs_roi_selected = dict()
    for i, roi in enumerate(bold_roi):
        func = partial(_compute_phaseshift_sliding_isc, data=roi, n_trs=n_trs, window_size=window_size,
                                                                            step_size=step_size,
                                                                            avg_over_roi=avg_over_roi, spatial=spatial,
                                                                            pairwise=pairwise,
                                                                            summary_statistic='median',
                                                                            n_shifts=n_shifts_batch,
                                                                            tolerate_nans=True,
                                                                            random_state=random_state)
        with ProcessPoolExecutor() as executor:
            iscs_roi_selected[roi_selected[i]] = list(tqdm(executor.map(unpack_and_call, [func]*len(kwargs), kwargs), total=len(kwargs)))

        with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois", 'wb') as f:
            pickle.dump(iscs_roi_selected, f)
        # kwargs = [{"data":roi for roi in bold_roi}]
        # with tqdm(total=n_shifts_batch) as pbar:
                # futures = [executor.submit(_compute_phaseshift_sliding_isc, bold, n_trs=n_trs, window_size=window_size,
                #                                                             step_size=step_size,
                #                                                             avg_over_roi=avg_over_roi, spatial=spatial,
                #                                                             pairwise=pairwise,
                #                                                             summary_statistic='median',
                #                                                             n_shifts=n_shifts_batch,
                #                                                             tolerate_nans=True,
                #                                                             random_state=random_state) for bold in bold_roi]
            
                # for future in as_completed(futures):
                #     iscs_roi_selected[roi_name] = future.result()
                #     pbar.update(1)
    
    # after parallelizing, the n batches are in n separate elements of phase_slide_isc; recombine them so that we have one nparray of observed, p, distribution for each roi
    x = {roi: [np.empty(shape=(n_windows, 1)), np.empty(shape=(n_windows, 1)), np.empty(shape=(n_shifts, n_windows, 1))] for roi in roi_selected} # init empty dict with appropriate shapes
    for roi in roi_selected:
        assert np.all(iscs_roi_selected[roi][0][0] == iscs_roi_selected[roi][1][0])  # make sure the "observed" is the same - should never change across batches
        assert np.all(iscs_roi_selected[roi][0][0] == iscs_roi_selected[roi][2][0])

        # joining the batched distributions
        dist = []
        for i in range(batch_size):  # number of loops = number of batches
            dist.append(iscs_roi_selected[roi][i][2])

        x[roi][0] = iscs_roi_selected[roi][0][0]  # take one of the "observed" ISC matrices since we asserted that they're all the same
        x[roi][2] = np.concatenate(dist)  # concatenate all n_shifts permutations
        x[roi][1] = p_from_null(x[roi][0], x[roi][2], side='two-sided', exact=False, axis=0)  # need to re-calculate p-values after concatenating permutations
    with open(f"{sliding_perm_path}_{n_shifts}perms_{len(roi_selected)}rois_x", 'wb') as f:
            pickle.dump(x, f)

if __name__ == '__main__':
    main()
err

slide_behav = np.load(f'{label_dir}/slide_behav_{task}_{smooth}.npy')
print(len(phase_slide_isc), len(slide_behav))

true_coefs = np.empty(shape=(len(roi_selected), len(emotions)))
true_means = []
true_stds = []
true_r2s = []
for r, roi in enumerate(roi_selected):
    true_isc_data = x[roi][0].flatten()  # observed, only need p and distribution for perm testing ISC
    alpha_range = np.logspace(-3, 3, 100)
    model = RidgeCV(alphas=alpha_range, store_cv_results=True)
    results = model.fit(slide_behav, true_isc_data)
    mses = results.cv_results_
    true_coefs[r] = results.coef_
    mean_mses = np.mean(mses, axis=0)
    std_mses = np.std(mses, axis=0)

    true_means.append(mean_mses[np.argmin(mean_mses)])
    true_stds.append(std_mses[np.argmin(std_mses)])
    true_r2s.append(model.score(slide_behav, true_isc_data))
[print(f"emotion consensus explains {true_r2s[r]:.2f} of variance in {roi} synchrony") for r, roi in enumerate(roi_selected)]

# if __name__ == '__main__':
#     main()
# plot null distribution
# plt.hist(phase_slide_isc['PCC'][2][:,:,-1])
# plt.show()

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

#
# # isc_vmpfc = iscs_roi_selected['vmPFC'].mean(axis=1)
# # scaled_isc = isc_vmpfc / np.max(isc_vmpfc)
# # slide_behav_scaled = slide_behav / np.max(slide_behav)
# #
# #
# # mask_img = np.load(f"{data_path}/../mask_img.npy")
# # ref_nii = nib.load(f"{data_path}/../ref_nii.nii.gz")
# # mask_coords = np.where(mask_img)
# # isc_img = np.full(ref_nii.shape, np.nan)
#
# # # save each sliding window as a Nifti file
# # for i in range(n_windows):
# #     if not os.path.exists(f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}'):
# #         isc_img[mask_coords] = iscs_roi_selected['wholebrain'][i]
# #         isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
# #         nib.save(isc_nii, f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}')
# #     else:
# #         print("Loading sliding ISC from file...")
# #         nib.load(f'{data_path}/sliding_isc/size{window_size}_step{step_size}_{str(i).zfill(2)}')
#
# # # correlate the ISC with the emotional report at each sliding window
# # corr_path = f'{data_path}/size{window_size}_step{step_size}.pkl'
# # if not os.path.exists(corr_path):
# #     corrs = np.empty((len(emotions), len(roi_selected)))
# #     for e in emotions:
# #         for roi in roi_selected:
# #             corr = pearsonr(slide_behav[:, emotions.index(e)], iscs_roi_selected[roi].mean(axis=1))
# #             corrs[emotions.index(e), roi_selected.index(roi)] = corr[0]
# #     with open(corr_path, 'wb') as f:
# #         pickle.dump(corrs, f)
# # else:
# #     with open(corr_path, 'rb') as f:
# #         corrs = pickle.load(f)
# #     print("Correlations loaded from file.")
#
# # n_perm = 10000
# #
# # # perform permutation testing per ROI
# # perm = np.empty(shape=(slide_behav.shape[1], len(roi_selected), n_perm, 2))  # emotions, rois, n_perm, r/p values
# # perm_path = f'{data_path}/sliding_isc/permutations/size{window_size}_step{step_size}_{n_perm}'
# # if not os.path.exists(perm_path):  # only compute if file DNE
# #     rng = np.random.default_rng()  # for rng.permutation
# #     for e in tqdm(range(slide_behav.shape[1]), position=0):  # number of emotions
# #         for r, roi in (rbar := tqdm(enumerate(roi_selected), leave=False)):
# #             rbar.set_description(f"Roi {r}")
# #             for i in range(n_perm):
# #                 perm[e, r, i] = pearsonr(iscs_roi_selected[roi].mean(axis=1),
# #                                          rng.permutation(slide_behav[:, e]))
# #     # save perm to pickle
# #     with open(perm_path, 'wb') as f:
# #         pickle.dump(perm, f)
# # else:
# #     with open(perm_path, 'rb') as f:
# #         perm = pickle.load(f)
# #     print("Permutation results loaded from file.")
# #
# # p_val = np.empty(shape=(len(emotions), len(roi_selected)))
# # # using the null distribution, get the p-values per roi and emotion
# # for e, _ in enumerate(emotions):
# #     for r, _ in enumerate(roi_selected):
# #         if corrs[e, r] > 0:
# #             p_val[e, r] = np.sum(perm[e, r, :, 0] >= corrs[e, r]) / n_perm
# #         else:
# #             p_val[e, r] = np.sum(perm[e, r, :, 0] <= corrs[e, r]) / n_perm
# #
# # # to choose the threshold, use full Bonferroni correction for number of rois
# # # at a 0.05 level, 0.05 / number of rois is the threshold and the p-values that are below this are significant
# # thresh = 0.05 / len(roi_selected)
# #
# # sig = p_val[:]
# # sig[sig > thresh] = np.nan
# #
# # # add labels to rows and columns
# # sig = pd.DataFrame(sig, index=emotions, columns=roi_selected)
# #
# # p_val = np.empty(shape=(len(emotions), len(roi_selected)))
# # for e, _ in enumerate(emotions):
# #     for r, _ in enumerate(roi_selected):
# #         if corrs[e, r] > 0:
# #             p_val[e, r] = np.sum(perm[e, r, :, 0] >= corrs[e, r]) / n_perm
# #         else:
# #             p_val[e, r] = np.sum(perm[e, r, :, 0] <= corrs[e, r]) / n_perm
# #
# # # to choose the threshold, use full Bonferroni correction for number of rois
# # # at a 0.05 level, 0.05 / number of rois is the threshold and the p-values that are below this are significant
# # thresh = 0.05 / len(roi_selected)
# #
# # sig = p_val[:]
# # sig[sig > thresh] = np.nan
# #
# # # add labels to rows and columns
# # sig = pd.DataFrame(sig, index=emotions, columns=roi_selected)
# #
# # # add a star to significant values on the plot
# # plt.figure()
# # for i, e in enumerate(emotions):
# #     plt.bar(np.arange(len(roi_selected)) + i * 0.2, corrs[i, :], width=0.2, label=e, color=['red', 'blue', 'purple', 'dimgray'][i])
# #     for j, r in enumerate(roi_selected):
# #         if not np.isnan(sig.iloc[i, j]):
# #             if corrs[i, j] > 0:
# #                 plt.text(j + i * 0.2, corrs[i, j] + 0.01, '*', color='k', fontsize=12, ha='center', va='center',
# #                          fontweight='bold')
# #             else:
# #                 plt.text(j + i * 0.2, corrs[i, j] - 0.04, '*', color='k', fontsize=12, ha='center', va='center',
# #                          fontweight='bold')
# # plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# # plt.ylabel("Pearson's r")
# # plt.legend(['Positive', 'Negative', 'Mixed', 'Neutral'])
# # plt.title(f"Correlation between sliding window ISC and feeling consensus")
# # plt.show()
# #
# # # add a star to significant values on the plot
# # plt.figure()
# # for i, e in enumerate(emotions[1:]):
# #     plt.bar(np.arange(len(roi_selected)) + i * 0.2, corrs[i+1, :], width=0.2, label=e)
# #     for j, r in enumerate(roi_selected):
# #         if not np.isnan(sig.iloc[i+1, j]):
# #             if corrs[i+1, j] > 0:
# #                 plt.text(j + i * 0.2, corrs[i+1, j] + 0.01, '*', color='k', fontsize=12, ha='center', va='center',
# #                          fontweight='bold')
# #             else:
# #                 plt.text(j + i * 0.2, corrs[i+1, j] - 0.04, '*', color='k', fontsize=12, ha='center', va='center',
# #                          fontweight='bold')
# # plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# # plt.ylabel("Pearson's r")
# # plt.legend(['Univalent', 'Mixed', 'Neutral'])
# # plt.title(f"Correlation between sliding window ISC and emotional consensus")
# # plt.savefig(f"{figure_path}/sliding_size{window_size}_step{step_size}")
# # plt.show()
# #
# # # plot emotion data by itself
# # plt.figure()
# # for i, e in enumerate(emotions):
# #     plt.plot(slide_behav[:, i], label=e)
# # plt.legend(emotions)
# # plt.title("Emotional consensus over time")
# # plt.show()
#
# # create correlation matrix for all emotions, showing only one half and listing the value of the correlation in the square
# # all_emo = np.load(f'{label_dir}/counts_{task}.npy')
# # mat = np.corrcoef(all_emo.T)
# # mat = np.tril(mat)
# # plt.figure()
# # plt.imshow(mat, cmap='coolwarm')
# # for i in range(mat.shape[0]):
# #     for j in range(mat.shape[1]):
# #         if i >= j:
# #             plt.text(j, i, f"{mat[i, j]:.2f}", ha='center', va='center')
# # plt.colorbar()
# # plt.xticks(range(len(emotions)), ['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
# # plt.yticks(range(len(emotions)), ['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
# # plt.title("Correlation matrix of emotion consensus")
# # # plt.savefig(f"{figure_path}/consensus_corr_matrix")
# # plt.show()
#
# # import statsmodels.api as sm
# #
# # betas = np.empty(shape=(len(roi_selected), len(emotions)))
# # pvalues = np.empty(shape=(len(roi_selected), len(emotions)))
# # for r, roi in enumerate(roi_selected):
# #     model = sm.OLS(iscs_roi_selected[roi].mean(axis=1), slide_behav[:, :4])
# #     results = model.fit()
# #     betas[r] = results.params
# #     pvalues[r] = results.pvalues
# # # print roi, emotion,  coef and pvalue if pvalue < 0.05
# # for r, roi in enumerate(roi_selected):
# #     for e, emo in enumerate(emotions):
# #         if pvalues[r, e] < 0.05:
# #             print(f"{roi}, {emo}: beta = {betas[r, e]:.3f}, p = {pvalues[r, e]:.3f}")
# #
# #
# # model = sm.OLS(iscs_roi_selected[roi].mean(axis=1), slide_behav[:, :4])
# # results = model.fit_regularized(alpha=1.0, L1_wt=0.0)
# # results.summary()
#
# # add labels for roi
# # plt.figure()
# # for i, e in enumerate(emotions):
# #     plt.bar(np.arange(len(roi_selected)) + i * 0.2, betas[:, i], width=0.2, label=e)
# # plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# # plt.ylabel("Beta")
# # plt.legend(emotions)
# # plt.title(f"Betas for each ROI")
# # plt.show()
#

#
perm_coefs = np.empty(shape=(len(roi_selected), n_shifts, len(emotions)))
perm_means = []
perm_stds = []
perm_r2s = np.empty(shape=(len(roi_selected), n_shifts))
for r, roi in enumerate(roi_selected):
    coefs = []
    means = []
    stds = []
    r2s = []
    for perm in range(n_shifts):
        shifted_isc_data = x[roi][-1][perm].flatten()  # observed, only need p and distribution for perm testing ISC
        alpha_range = np.logspace(-3, 3, 100)
        model = RidgeCV(alphas=alpha_range, store_cv_results=True)
        results = model.fit(slide_behav, shifted_isc_data)
        mses = results.cv_results_  # contains mean squared error for each alpha
        mean_mses = np.mean(mses, axis=0)
        std_mses = np.std(mses, axis=0)

        coefs.append(results.coef_)
        means.append(mean_mses[np.argmin(mean_mses)])
        stds.append(std_mses[np.argmin(std_mses)])
        r2s.append(model.score(slide_behav, shifted_isc_data))

    perm_coefs[r] = coefs
    perm_means.append(means)
    perm_stds.append(stds)
    perm_r2s[r] = r2s

# from isc_standalone import p_from_null
#
p_coef = np.empty(shape=(len(roi_selected), len(emotions)))
p_mean = []
p_std = []
p_r2 = np.empty(shape=(len(roi_selected)))
for r, roi in enumerate(roi_selected):
    for e, emo in enumerate(emotions):
        p_coef[r][e] = (p_from_null(true_coefs[r, e], perm_coefs[r,:,e]))
        # p_mean.append(p_from_null(true_means[r], perm_means[r]))
        # p_std.append(p_from_null(true_stds[r], perm_stds[r]))
        p_r2[r] = (p_from_null(true_r2s[r], perm_r2s[r]))

    # p_coef.append(p_roi[roi])

    # p_coef.append(p_from_null(true_coefs[r], perm_coefs[r]))
    # p_mean.append(p_from_null(true_means[r], perm_means[r]))
    # p_std.append(p_from_null(true_stds[r], perm_stds[r]))
    # p_r2.append(p_from_null(true_r2s[r], perm_r2s[r]))

# print the rois that are significant at p < 0.05 for each metric along with the value of the metric, rounded to 2 decimal places
for r, roi in enumerate(roi_selected):
    for e, emo in enumerate(emotions):
        if p_coef[r][e] < 0.05/9:
            print(f"{roi}, {emo}: Coef = {true_coefs[r, e]:.2f}, p = {p_coef[r][e]:.2f}")
            # print(f"{roi}: Coef = {true_coefs[r]:.2f}, p = {p_coef[r]:.2f}")
    # if p_mean[r] < 0.05:
    #     print(f"{roi}: Mean = {true_means[r]:.2f}, p = {p_mean[r]:.2f}")
    # if p_std[r] < 0.05:
    #     print(f"{roi}: Std = {true_stds[r]:.2f}, p = {p_std[r]:.2f}")
    if p_r2[r] < 0.05/9:
        print(f"{roi}: R2 = {true_r2s[r]:.2f}, p = {p_r2[r]:.2f}")


# plot the true and permuted distributions for the coefs for each emotion, all on the same plot
# Plot the null distribution for the coefficients along with the actual values
plt.figure(figsize=(15, 10))

for r, roi in enumerate(roi_selected):
    for e, emo in enumerate(emotions):
        plt.subplot(len(roi_selected), len(emotions), r * len(emotions) + e + 1)
        plt.hist(perm_coefs[r, :, e], bins=30, color='blue', alpha=0.5, label='Null Distribution')
        plt.axvline(true_coefs[r, e], color='red', linestyle='dashed', linewidth=2, label='Actual Value')
        plt.title(f"{roi} - {emo}")
        plt.xlabel('Coefficient Value')
        plt.ylabel('Frequency')
        plt.legend()

plt.tight_layout()
plt.show()

# plot only for PCC
plt.figure(figsize=(15, 10))
for e, emo in enumerate(emotions):
    plt.subplot(1, len(emotions), e + 1)
    plt.hist(perm_coefs[5, :, e], bins=100, color='blue', alpha=0.5, label='Null Distribution')
    plt.axvline(true_coefs[5, e], color='red', linestyle='dashed', linewidth=2, label='Actual Value')
    plt.title(f"PCC - {emo}")
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.legend()

plt.show()



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Prepare the data for plotting
# data = []
# for r, roi in enumerate(roi_selected):
#     for e, emo in enumerate(emotions):
#         for coef in perm_coefs[r, :, e]:
#             data.append([roi, emo, coef])
#         data.append([roi, emo, true_coefs[r, e]])
#
# df = pd.DataFrame(data, columns=['ROI', 'Emotion', 'Coefficient'])
#
# # Plot the violin plots
# plt.figure(figsize=(15, 10))
# sns.violinplot(x='ROI', y='Coefficient', hue='Emotion', data=df, split=True, inner='quartile')
# plt.axhline(0, color='k', linestyle='--')
# plt.title('Violin plots of null distribution and actual coefficients')
# plt.xlabel('ROI')
# plt.ylabel('Coefficient Value')
# plt.legend(title='Emotion')
# plt.tight_layout()
# plt.show()



# plot the true and permuted distributions for the isc values for each emotion and roi
plt.figure()
for i, e in enumerate(emotions):
    for r, roi in enumerate(roi_selected):
        plt.hist(x[roi][0].flatten(), bins=30, color='blue', alpha=0.5)
        plt.hist(x[roi][2], bins=30, color='red', alpha=0.5)
        plt.title(f"{e} {roi} ISC")
        plt.show()



# all_shifted_isc_data = phase_slide_isc[roi][-1]
# plt.plot(true_isc_data, 'b')
# plt.plot(all_shifted_isc_data.reshape((55,n_shifts)), alpha=0.4)
# # plt.legend(emotions)
# plt.show()

# plot hist of null dist r2s vs real one
# plt.hist(true_r2s, bins=30, color='blue')
# plt.hist(perm_r2s, bins=30, color='red')
# plt.show()



# # plt.figure()
# # for i, e in enumerate(emotions):
# #     plt.bar(np.arange(len(roi_selected)) + i * 0.2, coefs[:, i], width=0.2, label=e, color=['red', 'blue', 'purple', 'dimgray', 'brown'][i])
# # plt.xticks(range(len(roi_selected)), roi_selected, rotation=20)
# # plt.ylabel("Beta value")
# # plt.legend(['Positive', 'Negative', 'Mixed', 'Neutral', 'Cry'])
# # plt.title("Emotion consensus vs. ISC")
# # plt.tight_layout()
# # plt.show()
#
# phase_randomize()
# coefs = []
# all_results = []
# for r, roi in enumerate(roi_selected):
#     model = RidgeCV(alphas=alpha_range, store_cv_results=True)
#     results = model.fit(slide_behav, iscs_roi_selected[roi])
#     all_results.append(results)
#     coefs.append(results.coef_)
