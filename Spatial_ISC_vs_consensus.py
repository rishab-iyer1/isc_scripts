# Compares spatial ISC for particular ROIs with emotion rating consensus across all subjects

import pickle
import os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np

isc_path = '/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/data/iscs_roi_dict.pkl'
fig_path = '/Volumes/BCI/Ambivalent_Affect/RishabISC/ISC/figures'

if os.path.exists(isc_path):
    with open(isc_path, 'rb') as f:
        iscs_roi_selected = pickle.load(f)

    # compute percentage agreement for each TR (this block modified from Anthony's feeling_trend.py)
    df = pd.read_excel('/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Label_Aggregate.xlsx', skiprows=1)
    # needs openpyxl

    # Compute the percentage of subjects in each state at each time point
    P_counts = df[df == 'P'].count(axis=1)
    N_counts = df[df == 'N'].count(axis=1)
    M_counts = df[df == 'M'].count(axis=1)
    X_counts = df[df == 'X'].count(axis=1)

    total_counts = P_counts + N_counts + M_counts + X_counts

    P_percentage = (P_counts / total_counts) * 100
    N_percentage = (N_counts / total_counts) * 100
    M_percentage = (M_counts / total_counts) * 100
    X_percentage = (X_counts / total_counts) * 100

    # window_size = 10  # in samples
    # P_percentage_smoothed = P_percentage.rolling(window_size, center=True).mean()
    # N_percentage_smoothed = N_percentage.rolling(window_size, center=True).mean()
    # M_percentage_smoothed = M_percentage.rolling(window_size, center=True).mean()

    # iscs_roi_selected['auditory'].shape is (20, 454); in order to compare we need to take the average of all subjects
    # so we have one ISC value per TR
    avg_isc_roi = dict()
    for roi in iscs_roi_selected.keys():
        avg_isc_roi[roi] = iscs_roi_selected[roi].mean(axis=0)

    # now using these averages, we can find the correlation using Pearson's r
    pos = dict()
    neg = dict()
    mix = dict()
    neu = dict()
    sig_pos = dict()
    sig_neg = dict()
    sig_mix = dict()
    sig_neu = dict()
    for roi in avg_isc_roi.keys():
        pos[roi] = scipy.stats.pearsonr(P_percentage, avg_isc_roi[roi])
        neg[roi] = scipy.stats.pearsonr(N_percentage, avg_isc_roi[roi])
        mix[roi] = scipy.stats.pearsonr(M_percentage, avg_isc_roi[roi])
        neu[roi] = scipy.stats.pearsonr(X_percentage, avg_isc_roi[roi])

        if pos[roi][1] <= 0.05:
            print(f'Positive significant in {roi}; r={pos[roi][0]:.3f}, p={pos[roi][1]:.3f}')
            sig_pos[roi] = pos[roi]
        if neg[roi][1] <= 0.05:
            print(f'Negative significant in {roi}; r={neg[roi][0]:.3f}, p={neg[roi][1]:.3f}')
            sig_neg[roi] = neg[roi]
        if mix[roi][1] <= 0.05:
            print(f'Mixed significant in {roi}; r={mix[roi][0]:.3f}, p={mix[roi][1]:.3f}')
            sig_mix[roi] = mix[roi]
        if neu[roi][1] <= 0.05:
            print(f'Neutral significant in {roi}; r={neu[roi][0]:.3f}, p={neu[roi][1]:.3f}')
            sig_neu[roi] = neu[roi]

    # plot the correlations for significant ROIs
    for roi in sig_pos.keys():
        plt.figure()
        plt.plot(P_percentage, avg_isc_roi[roi], '.')
        plt.xlabel('Percentage of subjects in positive state')
        plt.ylabel('Average ISC')
        m = np.polyfit(P_percentage, avg_isc_roi[roi], 1)
        plt.plot(P_percentage, m[0] * P_percentage + m[1], label=f'r={sig_pos[roi][0]:.3f}, p={sig_pos[roi][1]:.3f}')
        plt.legend()
        plt.title(f'{roi} ISC vs. positive state')
        plt.savefig(f'{fig_path}/{roi}_ISC_vs_positive_state.png')

    for roi in sig_neg.keys():
        plt.figure()
        plt.plot(N_percentage, avg_isc_roi[roi], '.')
        plt.xlabel('Percentage of subjects in negative state')
        plt.ylabel('Average ISC')
        m = np.polyfit(N_percentage, avg_isc_roi[roi], 1)
        plt.plot(N_percentage, m[0] * N_percentage + m[1], label=f'r={sig_neg[roi][0]:.3f}, p={sig_neg[roi][1]:.3f}')
        plt.legend()
        plt.title(f'{roi} ISC vs. negative state')
        plt.savefig(f'{fig_path}/{roi}_ISC_vs_negative_state.png')

    for roi in sig_mix.keys():
        plt.figure()
        plt.plot(M_percentage, avg_isc_roi[roi], '.')
        plt.xlabel('Percentage of subjects in mixed state')
        plt.ylabel('Average ISC')
        m = np.polyfit(M_percentage, avg_isc_roi[roi], 1)
        plt.plot(M_percentage, m[0] * M_percentage + m[1], label=f'r={sig_mix[roi][0]:.3f}, p={sig_mix[roi][1]:.3f}')
        plt.legend()
        plt.title(f'{roi} ISC vs. mixed state')
        plt.savefig(f'{fig_path}/{roi}_ISC_vs_mixed_state.png')

    for roi in sig_neu.keys():
        plt.figure()
        plt.plot(X_percentage, avg_isc_roi[roi], '.')
        plt.xlabel('Percentage of subjects in neutral state')
        plt.ylabel('Average ISC')
        m = np.polyfit(X_percentage, avg_isc_roi[roi], 1)
        plt.plot(X_percentage, m[0] * X_percentage + m[1], label=f'r={sig_neu[roi][0]:.3f}, p={sig_neu[roi][1]:.3f}')
        plt.legend()
        plt.title(f'{roi} ISC vs. neutral state')
        plt.savefig(f'{fig_path}/{roi}_ISC_vs_neutral_state.png')

else:
    print('ISC data is not in this directory.')
