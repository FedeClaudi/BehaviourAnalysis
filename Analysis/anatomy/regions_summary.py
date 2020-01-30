# %%
import sys
sys.path.append('./')
import os 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline


from utils import *
from fcutils.file_io.utils import listdir
from fcutils.plotting.utils import save_figure


# %%
def plot_count_per_region_summary(ch0, ch1, mouse, n_regions_to_plot=16, ylabel='# cells'):
    f, axarr = plt.subplots(1, 2, figsize=(20, 14), sharey=True)
    f.suptitle('Cell count per region for mouse: {}'.format(mouse), fontsize=16)


    sns.barplot(ch0.values[:n_regions_to_plot], ch0.index[:n_regions_to_plot],\
                ax=axarr[0], palette='summer',)
    sns.barplot(ch1.values[:n_regions_to_plot], ch1.index[:n_regions_to_plot],\
                ax=axarr[1], palette='autumn',)

    sns.despine(ax=axarr[0], offset=10, left=True, right=False)
    sns.despine(ax=axarr[1], offset=10, left=True, right=True)

    axarr[0].set(title='Channel 0', ylabel=ylabel)
    axarr[0].invert_xaxis()
    axarr[1].set(title='Channel 1', ylabel=ylabel)
    axarr[1]

    # for ax in axarr:
    #     ax.tick_params(axis='x', labelrotation=45)

    return f, axarr



# %%

# ---------------------------------------------------------------------------- #
#                         MAKE BAR PLOT FOR EACH MOUSE                         #
# ---------------------------------------------------------------------------- #

for mouse in get_mice():
    ch0_cells = get_cells_for_mouse(mouse, ch=0)
    ch1_cells = get_cells_for_mouse(mouse, ch=1)

    if ch0_cells is None or ch1_cells is None:
        continue

    print(mouse)
    ch0_summary = get_count_by_brain_region(ch0_cells)
    ch1_summary = get_count_by_brain_region(ch1_cells)

    # Plot total cell count summary
    f, _ = plot_count_per_region_summary(ch0_summary[1], ch1_summary[1], mouse)
    save_figure(f, os.path.join(cellfinder_out_dir, mouse+'_count'), svg=False)

    # Plot normalised cell count
    f, _ = plot_count_per_region_summary(ch0_summary[2], ch1_summary[2], mouse,  ylabel='normalised # cells')
    save_figure(f, os.path.join(cellfinder_out_dir, mouse+'_normalised'), svg=False)





# %%

# ---------------------------------------------------------------------------- #
#                       GET COUNT PER REGION FOR ALL MICE                      #
# ---------------------------------------------------------------------------- #
summary = {r:[] for r in all_regions_acronyms}
summary['mouse'] = []

do_channel_0 = False
do_channel_1 = True

summaries = []
for i, mouse in enumerate(get_mice()):
    ch0_cells = get_cells_for_mouse(mouse, ch=0)
    ch1_cells = get_cells_for_mouse(mouse, ch=1)
    if ch0_cells is None or ch1_cells is None:
        continue

    print(mouse)
    ch0_summary = get_count_by_brain_region(ch0_cells)
    ch1_summary = get_count_by_brain_region(ch1_cells)

    if do_channel_0:
        df = pd.DataFrame(ch0_summary[2])
        df.columns = [mouse+'_ch0']
        summaries.append(df)
   
    if do_channel_1:
        df = pd.DataFrame(ch1_summary[2])
        df.columns = [mouse+'_ch1']
        summaries.append(df)
   

summary = pd.concat(summaries, axis=1)
summary = summary.dropna(how='all')
# idx = summary.sum(axis=1).sort_values(ascending=False).index
# summary=summary[idx]
summary.to_hdf(regions_summary_filepath, key='hdf')

summary.loc[all_regions.keys()].plot(kind='bar')
summary.loc[all_regions.keys()]
# %%
