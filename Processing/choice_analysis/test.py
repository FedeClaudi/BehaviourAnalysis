
import sys
sys.path.append('./')

import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import matplotlib.mlab as mlab
import matplotlib as mpl
import matplotlib.pyplot as plt

from Processing.choice_analysis.chioces_visualiser import ChoicesVisualiser as chioce_data
from database.NewTablesDefinitions import *

from database.database_fetch import *

data = chioce_data(run=False)

# Get all binary outcomes for both experiments

asym_binary, asym_binary_by_session = data.get_experiment_binary_outcomes('PathInt2')
sym_binary, sym_binary_by_session = data.get_experiment_binary_outcomes('Square Maze')

lights_onoff_individuals = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ],
    [1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1, 0, 1, 1, 1, 0, 1, 1, np.nan, np.nan, np.nan],
    [1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan, np.nan ]
])

lights_off_individuals = np.array([
    [0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, np.nan, np.nan,  np.nan],
    [1, 1, 0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan ],
    [1, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1, 1, 0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
])

lights_on_individuals = np.array([
    [1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [0, 0, 1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan],
    [1, 1, 1, 1, 1, 1, 0, 1, np.nan, np.nan],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1,],
    [1, 1, 1, 1, 0, np.nan, np.nan, np.nan, np.nan, np.nan],
])


#415
tr1 = np.array([
    [0, 0, 0],
    [1, 1, 1]
])

#416
tr2 = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,],
    [0, 0, 1, 0, 0, np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,]
])
#421
tr3 = np.array([
    [1, 1, 0, 1, np.nan,np.nan,np.nan,np.nan,],
    [1, 1, 1, 1, 1, 1, 0, 1]
])
#419
tr4 = np.array([
    [1, 0, np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan,  np.nan, ],
    [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
])
#418
tr5 = np.array([
    [1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0]
])
# 



old_escapes = pd.DataFrame((AllTrips & "is_escape='true'" & "is_trial='true'" & "experiment_name='PathInt2'").fetch())
old_audio = pd.DataFrame((AllTrips & "is_escape='true'" & "is_trial='true'" & "experiment_name='PathInt2'" & "stim_type='audio'").fetch())
old_vis = pd.DataFrame((AllTrips & "is_escape='true'" & "is_trial='true'" & "experiment_name='PathInt2'" & "stim_type!='audio'").fetch())

old_escapes_sym = pd.DataFrame((AllTrips & "is_escape='true'" & "is_trial='true'" & "experiment_name!='PathInt2'").fetch())

# old_escapes_sym = pd.merge(old_escapes_sym1, old_escapes_sym2)


old_trials = [1 if 'Right' in a else 0 for a in old_escapes['escape_arm'].values]


old_trials_individuals = []
for rec in set(old_escapes['session_uid'].values):
    d = old_escapes.loc[old_escapes['session_uid']==rec]
    old_trials_individuals.append([1 if 'Right' in a else 0 for a in d['escape_arm'].values])

lengths = [len(i) for i in old_trials_individuals]

old_escapes_padded = np.full((len(lengths), max(lengths)), np.nan)
for i, esc in enumerate(old_trials_individuals):
    if len(esc) <= 2: continue
    old_escapes_padded[i, :len(esc)] = esc

old_trials_individuals_sym = []
for rec in set(old_escapes_sym['session_uid'].values):
    d = old_escapes_sym.loc[old_escapes_sym['session_uid']==rec]
    old_trials_individuals_sym.append([1 if 'Right' in a else 0 for a in d['escape_arm'].values])

lengths = [len(i) for i in old_trials_individuals_sym]

old_escapes_sym_padded = np.full((len(lengths), max(lengths)), np.nan)
for i, esc in enumerate(old_trials_individuals_sym):
    if len(esc) <= 2: continue
    old_escapes_sym_padded[i, :len(esc)] = esc





old_trials_individuals_audio = []
for rec in set(old_audio['session_uid'].values):
    d = old_audio.loc[old_audio['session_uid']==rec]
    old_trials_individuals_audio.append([1 if 'Right' in a else 0 for a in d['escape_arm'].values])

lengths = [len(i) for i in old_trials_individuals_audio]

old_escapes_audio_padded = np.full((len(lengths), max(lengths)), np.nan)
for i, esc in enumerate(old_trials_individuals_audio):
    if len(esc) <= 2: continue
    old_escapes_audio_padded[i, :len(esc)] = esc







old_trials_individuals_visual = []
for rec in set(old_vis['session_uid'].values):
    d = old_vis.loc[old_vis['session_uid']==rec]
    old_trials_individuals_visual.append([1 if 'Right' in a else 0 for a in d['escape_arm'].values])

lengths = [len(i) for i in old_trials_individuals_visual]

old_escapes_visual_padded = np.full((len(lengths), max(lengths)), np.nan)
for i, esc in enumerate(old_trials_individuals_visual):
    if len(esc) <= 2: continue
    old_escapes_visual_padded[i, :len(esc)] = esc






trials = [lights_off_individuals, lights_onoff_individuals, lights_on_individuals, old_escapes_padded, old_escapes_audio_padded, old_escapes_visual_padded, old_escapes_sym_padded]
names = [ 'OFF', 'ON-OFF', 'ON' , 'ON - old','ON - old AUDIO', 'ON - old LOOM', 'SYM -old']
colors = [[.2, .2, .6],[.8, .8, .6], [.6, .9, .6], [.4, .8, .4], [.4, .8, .4], [.4, .8, .4], [.4, .4, .4]]

f, ax = plt.subplots()

for i, tr in enumerate(trials):
    y = np.round(np.nanmean(tr, 1), 2)
    x = np.zeros(tr.shape[0])
    # pick a sigma and mu for normal distribution
    sigma = .1
    mu =0.01

    # generate normally distributed samples
    noise = sigma * np.random.randn(len(x)) + mu

    ax.scatter(x+i+noise, y, alpha=.9, s=80, c=colors[i])
    ax.scatter(i, np.nanmean(y), c='r', s=100)

x = np.arange(len(names))
ax.set(xticks=x, xticklabels=names)
plt.show()





