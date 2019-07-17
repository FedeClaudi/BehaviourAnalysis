# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

from statistics import mode
from scipy.signal import find_peaks as peaks

from Utilities.imports import *
from Utilities.import_tables import *

from Processing.plot.plot_distributions import plot_fitted_curve

from Processing.analyse_experiments import ExperimentsAnalyser

# %%
# Define class

class PsychometricAnalyser(ExperimentsAnalyser):
    maze_names = {"maze1":"asymmetric_long", "maze2":"asymmetric_mediumlong", "maze3":"asymmetric_mediumshort", "maze4":"symmetric"}

    def __init__(self):
        ExperimentsAnalyser.__init__(self)

        self.conditions = dict(
            maze1 =  self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
            maze2 =  self.get_sesions_trials(maze_design=2, naive=None, lights=1, escapes=True),
            maze3 =  self.get_sesions_trials(maze_design=3, naive=None, lights=1, escapes=True),
            maze4 =  self.get_sesions_trials(maze_design=4, naive=None, lights=1, escapes=True),
        )

        self.maze_names_r = {v:k for k,v in self.maze_names.items()}

    def get_paths_lengths(self, load=True):
        if not load:
            # Loop over each maze design and get the path lenght (from the tracking data)
            summary = dict(maze=[], left=[], right=[], ratio=[])
            for i in np.arange(4):
                mazen = i +1
                trials = self.get_sesions_trials(maze_design=mazen, naive=None, lights=None, escapes=True)

                # Get the length of each escape
                lengths = []
                for _, trial in trials.iterrows():
                    lengths.append(np.sum(calc_distance_between_points_in_a_vector_2d(trial.tracking_data[:, :2])))
                trials["lengths"] = lengths

                left_trials, right_trials = [t.trial_id for i,t in trials.iterrows() if "left" in t.escape_arm.lower()], [t.trial_id for i,t in trials.iterrows() if "right" in t.escape_arm.lower()]
                left, right = trials.loc[trials.trial_id.isin(left_trials)], trials.loc[trials.trial_id.isin(right_trials)]

                # Make dict for summary df
                l, r  = percentile_range(left.lengths, low=10).low, percentile_range(right.lengths, low=10).low
                summary["maze"].append(self.maze_names_r[self.maze_designs[mazen]])
                summary["left"].append(round(l, 2))
                summary["right"].append(round(r, 2))
                summary["ratio"].append(round(l / r, 4))

            self.paths_lengths = pd.DataFrame.from_dict(summary)
            self.paths_lengths.to_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))
        else:
            self.paths_lengths = pd.read_pickle(os.path.join(self.metadata_folder, "path_lengths.pkl"))

        geopaths = self.get_arms_lengths_with_agent(load=True)
        self.paths_lengths = pd.merge(self.paths_lengths, geopaths)
        short_arm_dist = self.paths_lengths.loc[self.paths_lengths.maze=="maze4"].distance.values[0]
        self.paths_lengths["georatio"] = [round(x / short_arm_dist, 4) for x in self.paths_lengths.distance.values]


    """
        ||||||||||||||||||||||||||||    GETTERS     |||||||||||||||||||||
    """
    def get_hb_modes(self):
        trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False)
        n_bins = 100
        bins = {k:[np.digitize(a, np.linspace(0, 1, n_bins)) for a in v.T] for k,v in trace.items()}
        modes = {k:[np.median(b)/n_bins for b in bins] for k,bins in bins.items()}

        return modes


    """
        ||||||||||||||||||||||||||||    PLOTTERS     |||||||||||||||||||||
    """
    def plot_pr_by_condition(self, raw_individuals=False, exclude_experiments=[None], ax=None):
        # Get paths length ratios and p(R) by condition
        self.get_paths_lengths()
        hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(self.conditions)
        
        # Get modes on individuals posteriors and grouped bayes
        modes = self.get_hb_modes()
        grouped_modes = self.bayes_by_condition_analytical(mode="grouped", plot=False) 

        # Plot each individual's pR and the group mean as a factor of L/R length ratio
        if ax is None: f, ax = plt.subplots()
        lr_ratios_mean_pr = {"grouped":[], "individuals_x":[], "individuals_y":[]}
        for i, (condition, pr) in enumerate(p_r.items()):
            x = self.paths_lengths.loc[self.paths_lengths.maze == condition].georatio.values
            if not raw_individuals:
                y = pr
            else:
                y = modes[condition]

            if condition not in exclude_experiments:# ? use the data for curves fitting
                k = .7
                lr_ratios_mean_pr["grouped"].append((x[0], np.median(y)))  
                lr_ratios_mean_pr["individuals_x"].append([x[0] for _ in np.arange(len(y))])
                lr_ratios_mean_pr["individuals_y"].append(y)
            else: 
                k = .1
                del grouped_modes[condition]
            ax.scatter(np.random.normal(x, 0.01, size=len(y)), y, alpha=.8, color=desaturate_color(self.colors[i+1], k=k), s=250)


          
        # Fit sigmoid to group data and individual mice
        plot_fitted_curve(sigmoid, np.array([x for x,y in lr_ratios_mean_pr["grouped"]]), np.array(list(grouped_modes.values())), [0.75, 1.5], ax, 
                                scatter_kwargs={"color":white, "alpha":1, "s":250}, 
                                line_kwargs={"color":desaturate_color(white), "lw":5, "label":"sigmoid means", "ls":"--"})
        plot_fitted_curve(sigmoid, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), [0.75, 1.5], ax,   # ? ind. sigmoid
                                scatter_kwargs={"color":white, "alpha":0, "s":250}, 
                                line_kwargs={"color":white, "lw":4, "label":"sigmoid individudals"})

        # Fit step function and linear function
        # plot_fitted_curve(step_function, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), [0.75, 1.5], ax, 
        #                         # fit_kwargs={"bounds":[(0, 0, 0), (1, 2, 1)]}, 
        #                         scatter_kwargs={"color":pink, "alpha":0, "s":250}, 
        #                         line_kwargs={"color":pink, "lw":4, "label":"step func individudals", "ls":"--"})

        # plot_fitted_curve(linear_func, np.hstack(lr_ratios_mean_pr["individuals_x"]), np.hstack(lr_ratios_mean_pr["individuals_y"]), [0.75, 1.5], ax, 
        #                         scatter_kwargs={"color":teal, "alpha":0, "s":250}, 
        #                         line_kwargs={"color":teal, "lw":4, "label":"linear func individudals", "ls":"--"})
        
        # Fix plotting
        ax.axhline(.5, color=grey, lw=2, ls="--", alpha=.2)
        ax.set(ylim=[-0.05, 1.05], ylabel="p(R)", title="p(R) per mouse per maze", xlabel="L/R length raito",
                 xticks = self.paths_lengths.georatio.values, xticklabels = self.paths_lengths.georatio.values)
        ax.legend()


    def plot_heirarchical_bayes_effect(self):
        # Get hierarchical Bayes modes and individual mice p(R)
        hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(self.conditions)
        trace = self.bayes_by_condition(conditions=self.conditions, load=True, tracefile="psychometric_individual_bayes.pkl", plot=False) 

        # Get the mode of the posteriors
        n_bins = 100
        bins = {k:[np.digitize(a, np.linspace(0, 1, n_bins)) for a in v.T] for k,v in trace.items()}
        modes = {k:[np.median(b)/n_bins for b in bins] for k,bins in bins.items()}

        f, axarr = plt.subplots(ncols=4, sharex=True, sharey=True)
        
        for i, (exp, ax) in enumerate(zip(trace.keys(), axarr)):
            ax.scatter(np.random.normal(0, .025, size=len(p_r[exp])), p_r[exp], color=self.colors[i+1], alpha=.5, s=200)
            ax.scatter(np.random.normal(1, .025, size=len(modes[exp])), modes[exp], color=self.colors[i+1], alpha=.5, s=200)

            ax.scatter(0, np.mean(p_r[exp]), color="w", alpha=1, s=300)
            ax.scatter(1, np.mean(modes[exp]), color="w", alpha=1, s=300)

            ax.set(title=exp, xlim=[-.1, 1.1], ylim=[-.02, 1.02], xticks=[0, 1], xticklabels=["Raw", "Bayes"], ylabel="p(R)")


        
if __name__ == "__main__":
    pa = PsychometricAnalyser()

    f, axarr = plt.subplots(figsize=large_square_fig, ncols=3, nrows=2)
    axarr = axarr.flatten()
    pa.plot_pr_by_condition(raw_individuals=True, ax=axarr[0])
    for i, exp in enumerate(pa.conditions.keys()):
        pa.plot_pr_by_condition(raw_individuals=True, exclude_experiments= [exp], ax=axarr[i+1])

    plt.show()



#%%
