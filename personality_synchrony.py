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
cry = df['Did you cry during either video? If so, at what point in the videos and stories did you cry? Try to be as '
         'specific as possible as to when you began crying.']
df = df[['I see myself as: - Extraverted, enthusiastic.', 'I see myself as: - Critical, quarrelsome.',
         'I see myself as: - Dependable, self-disciplined.', 'I see myself as: - Anxious, easily upset.',
         'I see myself as: - Open to new experiences, complex', 'I see myself as: - Reserved, quiet',
         'I see myself as: - Sympathetic, warm', 'I see myself as: - Disorganized, careless',
         'I see myself as: - Calm, emotionally stable', 'I see myself as: - Conventional, uncreative']]

# TIPI scale scoring (“R” denotes reverse-scored items):
# Extraversion: 1, 6R; Agreeableness: 2R, 7; Conscientiousness; 3, 8R; Emotional Stability: 4R, 9;
# Openness to Experiences: 5, 10R.

code = ['Disagree strongly', 'Disagree moderately', 'Disagree a little', 'Neither agree nor disagree', 'Agree a little',
        'Agree moderately', 'Agree strongly']

to_reverse = df.iloc[:, [1, 3, 5, 7, 9]]
to_forward = df.iloc[:, [0, 2, 4, 6, 8]]
to_reverse = to_reverse.replace(code, range(7, 0, -1))
to_forward = to_forward.replace(code, range(1, 8, 1))
recoded = pd.concat([to_forward, to_reverse], axis=1)
recoded = recoded[['I see myself as: - Extraverted, enthusiastic.', 'I see myself as: - Critical, quarrelsome.',
                   'I see myself as: - Dependable, self-disciplined.', 'I see myself as: - Anxious, easily upset.',
                   'I see myself as: - Open to new experiences, complex', 'I see myself as: - Reserved, quiet',
                   'I see myself as: - Sympathetic, warm', 'I see myself as: - Disorganized, careless',
                   'I see myself as: - Calm, emotionally stable', 'I see myself as: - Conventional, uncreative']]

composite = pd.DataFrame()
composite['Extraversion'] = recoded[recoded.columns[0]] + recoded[df.columns[5]] / 2
composite['Agreeableness'] = recoded[recoded.columns[1]] + recoded[df.columns[6]] / 2
composite['Conscientiousness'] = recoded[recoded.columns[2]] + recoded[df.columns[7]] / 2
composite['Emotional Stability'] = recoded[recoded.columns[3]] + recoded[df.columns[8]] / 2
composite['Openness to Experiences'] = recoded[recoded.columns[4]] + recoded[df.columns[9]] / 2

# for each pair, calculate the Euclidean distance between their personality scores
# form into a dictionary where keys are tuples of strings of subject ids
pers = {}
for pair in itertools.combinations(df.index, 2):
    pers[pair] = math.sqrt(np.sum((composite.loc[pair[0]] - composite.loc[pair[1]]) ** 2))

# pers = pd.DataFrame()
# for i, j in itertools.combinations(composite.index, 2):
#     pers.loc[i, j] = np.linalg.norm(composite.loc[i] - composite.loc[j])

# read in the pairwise temporal ISC data
with open('../data/isc_temporal_pairwise_n27_roi7.pkl', 'rb') as f:
    temporal = pickle.load(f)

subj_mapping = [x for x in itertools.combinations(list(df.index), 2)]
wb = temporal['wholebrain']
avg_z = np.array(np.tanh(np.mean(np.arctanh(wb), axis=1))).T
isc = {subj_mapping[i]: avg_z[i] for i in range(len(subj_mapping))}

# create a list of tuples of each combination of subjects we have personality data for
pers_ids = list(itertools.combinations(list(df.index), 2))

# find intersection between pers_ids and subj_mapping
common = [x for x in pers_ids if x in subj_mapping]

# subset isc to the same keys as pers
isc = {k: v for k, v in isc.items() if k in common}
pers = {k: v for k, v in pers.items() if k in common}

to_corr = np.array([list(isc.values()), list(pers.values())], dtype=object)

# correlation between the two rows of to_corr
print(pearsonr(to_corr[0], to_corr[1]))
