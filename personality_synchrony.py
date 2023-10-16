import pandas as pd
import numpy as np
import itertools
import pickle
import math
from scipy.stats import pearsonr

# read subject ID as index
df = pd.read_csv('../data/VR+Film_September+26%2C+2023_15.39.csv', header=1, skiprows=[2], index_col=0)
df = df.iloc[[0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 23]]
# change order of columns
cry = df['Did you cry during either video? If so, at what point in the videos and stories did you cry? Try to be as specific as possible as to when you began crying.']
df = df[['I see myself as: - Extraverted, enthusiastic.', 'I see myself as: - Critical, quarrelsome.', 'I see myself as: - Dependable, self-disciplined.', 'I see myself as: - Anxious, easily upset.', 'I see myself as: - Open to new experiences, complex', 'I see myself as: - Reserved, quiet', 'I see myself as: - Sympathetic, warm', 'I see myself as: - Disorganized, careless', 'I see myself as: - Calm, emotionally stable', 'I see myself as: - Conventional, uncreative']]

# TIPI scale scoring (“R” denotes reverse-scored items):
# Extraversion: 1, 6R; Agreeableness: 2R, 7; Conscientiousness; 3, 8R; Emotional Stability: 4R, 9;
# Openness to Experiences: 5, 10R.

code = ['Disagree strongly', 'Disagree moderately', 'Disagree a little', 'Neither agree nor disagree', 'Agree a little', 'Agree moderately', 'Agree strongly']

df.iloc[:, [1, 3, 5, 7, 9]].replace(code, range(7, 0, -1), inplace=True)
df.iloc[:, [0, 2, 4, 6, 8]].replace(code, range(1, 8, 1), inplace=True)
to_reverse = df.iloc[:, [1, 3, 5, 7, 9]]
to_forward = df.iloc[:, [0, 2, 4, 6, 8]]
to_reverse.replace(code, range(7, 0, -1), inplace=True)
to_forward.replace(code, range(1, 8, 1), inplace=True)
recoded = pd.concat([to_forward, to_reverse], axis=1)
recoded = recoded[['I see myself as: - Extraverted, enthusiastic.', 'I see myself as: - Critical, quarrelsome.',
                   'I see myself as: - Dependable, self-disciplined.', 'I see myself as: - Anxious, easily upset.', 'I see myself as: - Open to new experiences, complex', 'I see myself as: - Reserved, quiet', 'I see myself as: - Sympathetic, warm', 'I see myself as: - Disorganized, careless', 'I see myself as: - Calm, emotionally stable', 'I see myself as: - Conventional, uncreative']]

composite = pd.DataFrame()
composite['Extraversion'] = recoded[recoded.columns[0]] + recoded[df.columns[5]] / 2
composite['Agreeableness'] = recoded[recoded.columns[1]] + recoded[df.columns[6]] / 2
composite['Conscientiousness'] = recoded[recoded.columns[2]] + recoded[df.columns[7]] / 2
composite['Emotional Stability'] = recoded[recoded.columns[3]] + recoded[df.columns[8]] / 2
composite['Openness to Experiences'] = recoded[recoded.columns[4]] + recoded[df.columns[9]] / 2

# for each pair, calculate the euclidean distance between their personality scores
# form into a dictionary where keys are tuples of strings of subject ids
pers = {}
for pair in itertools.combinations(df.index, 2):
    pers[pair] = math.sqrt(sum((composite.loc[pair[0]] - composite.loc[pair[1]]) ** 2))

# pers = pd.DataFrame()
# for i, j in itertools.combinations(composite.index, 2):
#     pers.loc[i, j] = np.linalg.norm(composite.loc[i] - composite.loc[j])

# read in the pairwise temporal ISC data
with open('../data/isc_temporal_pairwise_n20_roi7.pkl', 'rb') as f:
    temporal = pickle.load(f)

subj_mapping = [('P2', 'P3'), ('P2', 'P1'), ('P2', 'N2'), ('P2', 'N1'), ('P2', 'N3'), ('P2', 'VR14'), ('P2', 'VR4'), ('P2', 'VR13'), ('P2', 'VR5'), ('P2', 'VR8'), ('P2', 'VR9'), ('P2', 'VR7'), ('P2', 'VR12'), ('P2', 'VR1'), ('P2', 'VR16'), ('P2', 'VR2'), ('P2', 'VR10'), ('P2', 'VR11'), ('P2', 'VR3'), ('P3', 'P1'), ('P3', 'N2'), ('P3', 'N1'), ('P3', 'N3'), ('P3', 'VR14'), ('P3', 'VR4'), ('P3', 'VR13'), ('P3', 'VR5'), ('P3', 'VR8'), ('P3', 'VR9'), ('P3', 'VR7'), ('P3', 'VR12'), ('P3', 'VR1'), ('P3', 'VR16'), ('P3', 'VR2'), ('P3', 'VR10'), ('P3', 'VR11'), ('P3', 'VR3'), ('P1', 'N2'), ('P1', 'N1'), ('P1', 'N3'), ('P1', 'VR14'), ('P1', 'VR4'), ('P1', 'VR13'), ('P1', 'VR5'), ('P1', 'VR8'), ('P1', 'VR9'), ('P1', 'VR7'), ('P1', 'VR12'), ('P1', 'VR1'), ('P1', 'VR16'), ('P1', 'VR2'), ('P1', 'VR10'), ('P1', 'VR11'), ('P1', 'VR3'), ('N2', 'N1'), ('N2', 'N3'), ('N2', 'VR14'), ('N2', 'VR4'), ('N2', 'VR13'), ('N2', 'VR5'), ('N2', 'VR8'), ('N2', 'VR9'), ('N2', 'VR7'), ('N2', 'VR12'), ('N2', 'VR1'), ('N2', 'VR16'), ('N2', 'VR2'), ('N2', 'VR10'), ('N2', 'VR11'), ('N2', 'VR3'), ('N1', 'N3'), ('N1', 'VR14'), ('N1', 'VR4'), ('N1', 'VR13'), ('N1', 'VR5'), ('N1', 'VR8'), ('N1', 'VR9'), ('N1', 'VR7'), ('N1', 'VR12'), ('N1', 'VR1'), ('N1', 'VR16'), ('N1', 'VR2'), ('N1', 'VR10'), ('N1', 'VR11'), ('N1', 'VR3'), ('N3', 'VR14'), ('N3', 'VR4'), ('N3', 'VR13'), ('N3', 'VR5'), ('N3', 'VR8'), ('N3', 'VR9'), ('N3', 'VR7'), ('N3', 'VR12'), ('N3', 'VR1'), ('N3', 'VR16'), ('N3', 'VR2'), ('N3', 'VR10'), ('N3', 'VR11'), ('N3', 'VR3'), ('VR14', 'VR4'), ('VR14', 'VR13'), ('VR14', 'VR5'), ('VR14', 'VR8'), ('VR14', 'VR9'), ('VR14', 'VR7'), ('VR14', 'VR12'), ('VR14', 'VR1'), ('VR14', 'VR16'), ('VR14', 'VR2'), ('VR14', 'VR10'), ('VR14', 'VR11'), ('VR14', 'VR3'), ('VR4', 'VR13'), ('VR4', 'VR5'), ('VR4', 'VR8'), ('VR4', 'VR9'), ('VR4', 'VR7'), ('VR4', 'VR12'), ('VR4', 'VR1'), ('VR4', 'VR16'), ('VR4', 'VR2'), ('VR4', 'VR10'), ('VR4', 'VR11'), ('VR4', 'VR3'), ('VR13', 'VR5'), ('VR13', 'VR8'), ('VR13', 'VR9'), ('VR13', 'VR7'), ('VR13', 'VR12'), ('VR13', 'VR1'), ('VR13', 'VR16'), ('VR13', 'VR2'), ('VR13', 'VR10'), ('VR13', 'VR11'), ('VR13', 'VR3'), ('VR5', 'VR8'), ('VR5', 'VR9'), ('VR5', 'VR7'), ('VR5', 'VR12'), ('VR5', 'VR1'), ('VR5', 'VR16'), ('VR5', 'VR2'), ('VR5', 'VR10'), ('VR5', 'VR11'), ('VR5', 'VR3'), ('VR8', 'VR9'), ('VR8', 'VR7'), ('VR8', 'VR12'), ('VR8', 'VR1'), ('VR8', 'VR16'), ('VR8', 'VR2'), ('VR8', 'VR10'), ('VR8', 'VR11'), ('VR8', 'VR3'), ('VR9', 'VR7'), ('VR9', 'VR12'), ('VR9', 'VR1'), ('VR9', 'VR16'), ('VR9', 'VR2'), ('VR9', 'VR10'), ('VR9', 'VR11'), ('VR9', 'VR3'), ('VR7', 'VR12'), ('VR7', 'VR1'), ('VR7', 'VR16'), ('VR7', 'VR2'), ('VR7', 'VR10'), ('VR7', 'VR11'), ('VR7', 'VR3'), ('VR12', 'VR1'), ('VR12', 'VR16'), ('VR12', 'VR2'), ('VR12', 'VR10'), ('VR12', 'VR11'), ('VR12', 'VR3'), ('VR1', 'VR16'), ('VR1', 'VR2'), ('VR1', 'VR10'), ('VR1', 'VR11'), ('VR1', 'VR3'), ('VR16', 'VR2'), ('VR16', 'VR10'), ('VR16', 'VR11'), ('VR16', 'VR3'), ('VR2', 'VR10'), ('VR2', 'VR11'), ('VR2', 'VR3'), ('VR10', 'VR11'), ('VR10', 'VR3'), ('VR11', 'VR3')]
wb = temporal['auditory']
avg_z = np.array(np.tanh(np.mean(np.arctanh(wb), axis=1))).T
isc = {subj_mapping[i]: avg_z[i] for i in range(len(subj_mapping))}

# create a list of tuples of each combination of subjects we have personality data for
pers_ids = list(itertools.combinations(list(df.index), 2))

# find intersection between pers_ids and subj_mapping
common = [x for x in pers_ids if x in subj_mapping]

# subset isc to the same keys as pers
isc = {k: v for k, v in isc.items() if k in common}
pers = {k: v for k, v in pers.items() if k in common}

to_corr = np.array([list(isc.values()), list(pers.values())])

# correlation between the two rows of to_corr
print(pearsonr(to_corr[0], to_corr[1]))
