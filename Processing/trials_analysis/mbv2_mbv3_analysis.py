# %%
from Utilities.imports import *
%matplotlib inline  

# %%
filepath = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/MBV2_MBV3_trials.xlsx"

data = pd.read_excel(filepath)
# data = data.sort_values("naive")[::-1]
data

data = data.loc[data.version == 2]


#%%
# organise the trial data in a minigful way

n_trials = [len(d.trials) for i,d in data.iterrows()]
sort_index_n_trials = np.argsort(n_trials)
n_baseline_trials = data.first_close.values
max_baseline = np.max(n_baseline_trials)
arr_len = max_baseline + np.max(n_trials)

align_shift = [max_baseline - nb for nb in n_baseline_trials]

trials = np.zeros((len(data), arr_len))

for i, d in data.iterrows():
    trials_as_array = np.array([1 if t=='A' else 2 if t=="B" else 1.5 for t in d.trials])
    trials[i, align_shift[i]:align_shift[i]+n_trials[i]] = trials_as_array

trials = trials[sort_index_n_trials[::-1]]

# Sort by naive vs not naive
naives = [d.naive for i,d in data.iterrows()]
n_naives = naives.count("Y")

naives = np.array(naives)[sort_index_n_trials[::-1]]
sort_idx = np.argsort(naives)[::-1]

# get prob of B at every trials
_trials = np.full_like(trials, np.nan)
_trials[(trials==1)|(trials==1.5)] = 0
_trials[trials==2] = 1
b_prob = np.nanmean(_trials, 0)

#%%
f, axarr = plt.subplots(ncols=1, nrows=2, figsize=(8, 10), sharex="col")
axarr = axarr.flatten()
axarr[0].imshow(trials[sort_idx], cmap="hot")
axarr[0].axvline(max_baseline-.5, color="b", lw=3)
axarr[0].axhline(n_naives-.5, color="g", lw=3)

axarr[0].set(xlabel="trials", ylabel="mice")

axarr[1].bar(np.linspace(0, trials.shape[1], trials.shape[1]), np.nanmean(_trials, 0))
axarr[1].axvline(max_baseline-.5, color="b", lw=3)

# axarr[1].bar(np.linspace(0, trials.shape[0], trials.shape[0]), np.nanmean(_trials, 1))

#%%


#%%
