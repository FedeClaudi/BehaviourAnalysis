import sys
sys.path.append('./')

from scipy.signal import resample

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

print("\n\n\n")

class Torosity(Trials):
    def __init__(self, mtype):
        Trials.__init__(self, exp_1_mode=True, just_escapes="all")

        self.trials = self.trials.loc[self.trials['grouped_experiment_name']==mtype]  # only keep the trials from asym exp

        # Create scaled agent
        self.scale_factor = 0.25

        if mtype == "asymmetric":
            self.agent = GradientAgent(
                                        maze_type = "asymmetric_large",
                                        maze_design = "PathInt2.png",
                                        grid_size = int(1000*self.scale_factor), 
                                        start_loc= [int(500*self.scale_factor), int(800*self.scale_factor)], 
                                        goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)
        else:
            self.agent = GradientAgent(
                                        maze_type = "asymmetric_large",
                                        maze_design = "Square Maze.png",
                                        grid_size = int(1000*self.scale_factor), 
                                        start_loc= [int(500*self.scale_factor), int(730*self.scale_factor)], 
                                        goal_loc = [int(500*self.scale_factor), int(260*self.scale_factor)], stride=1)

        scaled_blocks = {}
        for k, states in self.agent.bridges_block_states.items():
            scaled_blocks[k] = [(int(x*self.scale_factor), int(y*self.scale_factor)) for x,y in states]
        self.agent.bridges_block_states = scaled_blocks

        # lookup vars
        self.results_keys = ["walk_distance", "tracking_distance", "torosity", "tracking_data", "escape_arm", "is_escape", "binned_torosity"]

        if mtype == "asymmetric":
            self.bridges_lookup = dict(Right_Medium="right", Left_Far="left")
        else:
            self.bridges_lookup = dict(Right_Medium="right", Left_Medium="left")

        self.results_path = "Processing\\trials_analysis\\torosity_results_toshelter_{}.pkl".format(mtype)
        self.plots_save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\torosity\\check"

    """
        #########################################################################################################################################################
                UTILS
        #########################################################################################################################################################
    """

    def smallify_tracking(self, tracking):
        return np.multiply(tracking, self.scale_factor).astype(np.int32)

    @staticmethod
    def zscore_and_sort(res):
        res['walk_distance_z'] = stats.zscore(res.walk_distance)
        res['tracking_distance_z'] = stats.zscore(res.tracking_distance)
        res['torosity_z'] = stats.zscore(res.torosity)

        res.sort_values("torosity_z")

        return res

    """
        #########################################################################################################################################################
                PROCESSING
        #########################################################################################################################################################
    """

    def time_binned_torosity(self, tracking, br):
        binned_tor = namedtuple("bt", "i_start i_end torosity")
        self.agent._reset()

        window_size, window_step = 10, 5

        i_start = 0
        i_end = i_start + window_size

        binned_tor_list = []
        while i_end <= tracking.shape[0]:
            binned_tracking = tracking[i_start:i_end, :, :]
            i_start += window_step
            i_end += window_step

            torosity = self.process_one_trial(None, br, tracking=binned_tracking, goal=list(binned_tracking[-1, :2, 0]))

            binned_tor_list.append(binned_tor(i_start, i_end, torosity))

        return binned_tor_list

    def process_one_trial(self, trial, br, tracking=None, goal=None):
        # Reset 
        self.agent._reset()

        if tracking is None:
            # scale down the tracking data
            tracking = self.smallify_tracking(trial.tracking_data.astype(np.int16))

        # get the start and end of the escape
        self.agent.start_location = list(tracking[0, :2, 0])
        if goal is not None:
            self.agent.goal_location = goal
        else:
            self.agent.goal_location = list(tracking[-1, :2, 0])

        # get the new geodistance to the location where the escape ends
        self.agent.geodesic_distance = self.agent.get_geo_to_point(self.agent.goal_location)
        if self.agent.geodesic_distance is None and goal is None: return None, None, None
        elif self.agent.geodesic_distance is None and goal is not None: return np.nan

        # Introduce blocked bridge if LEFT escape
        if "left" in br.lower():
            self.agent.introduce_blockage("right_large", update =True)

        # do a walk with the same geod
        walk = np.array(self.agent.walk())


        # compute stuff
        walk_distance  = np.sum(calc_distance_between_points_in_a_vector_2d(walk)) 

        # if goal is None: # ? not very elegant, but it it is to say that we check only when we are processing the whole trial
        #     if len(walk < 10): raise ValueError("Why u not walking")
        #     if walk_distance < 1: raise ValueError("Something went wrong with the walk distance")

        tracking_distance   = np.sum(calc_distance_between_points_in_a_vector_2d(tracking[:, :2, 0])) 
        torosity = tracking_distance/walk_distance

        if (np.isnan(torosity) or np.isinf(torosity)) and goal is None: raise ValueError


        if trial is not None:
            # Create results
            results = dict(
                walk_distance       = walk_distance,
                tracking_distance   = tracking_distance,
                torosity = torosity,
                tracking_data = tracking,
                escape_arm = br,
                is_escape = trial.is_escape
            )

            return tracking, walk, results
        else:
            return torosity

    def analyse_all(self, plot=False, save_plots=False):
        print("processing all trials")
        all_res = {k:[] for k in self.results_keys}

        trials = [t for i,t in self.trials.iterrows()]
        for trial in tqdm(trials):
            try:
                tracking, walk, res = self.process_one_trial(trial, self.bridges_lookup[trial.escape_arm])
            except: 
                continue

            if tracking is None: continue

            res["binned_torosity"] = self.time_binned_torosity(tracking, self.bridges_lookup[trial.escape_arm])
            if plot:
                self.plot_time_binned_torosity(tracking, walk, res["binned_torosity"], res, save_plots=save_plots, trial_id=trial.trial_id)

            if res is None: continue

            for k,v in res.items(): all_res[k].append(v)
        results = pd.DataFrame.from_dict(all_res)

        # clean_up and save
        results = results.loc[results.torosity != np.inf]
        results.to_pickle(self.results_path)

    """
        #########################################################################################################################################################
                PLOTTING
        #########################################################################################################################################################
    """
    def plot_one_escape_and_its_walk(self):  # one per arm
        f, axarr = plt.subplots(ncols=2)
        colors = get_n_colors(5)

        for i in range(1):
            for ax,  br in zip(axarr,  self.bridges_lookup.keys()):
                # Get some tracking
                trials = self.trials.loc[self.trials['escape_arm']==br]
                trial = trials.iloc[np.random.randint(0, len(trials))]

                # Process
                tracking, walk, res = self.process_one_trial(trial, br)
               
                if tracking is None: continue
                # Plot
                _ = self.agent.plot_walk(walk, ax=ax)
                ax.scatter(tracking[:, 0, 0], tracking[:, 1, 0],  c='r', s=100, alpha=.25)

                ax.set(
                    title = "Walk: {} - Track: {} - Tor: {}".format(round(res['walk_distance'], 2), round(res['tracking_distance'], 2), round(res['torosity'], 2))
                )
                
    def plot_time_binned_torosity(self, tracking, walk, binned_tor, res, save_plots=False, trial_id=None):
        f, ax = plt.subplots(figsize=(16, 16))

        rearranged = np.zeros((len(binned_tor), 3))

        for i, (i_start, i_end, tor) in enumerate(binned_tor):
            rearranged[i, :2] = tracking[i_start, :2, 0]
            try:
                rearranged[i, -1] = tor
            except:
                raise ValueError(tor)

        self.agent.plot_walk(walk, alpha=.6,  ax=ax)
        ax.scatter(rearranged[:, 0], rearranged[:, 1], c=rearranged[:, 2],  s=100, alpha=.8, cmap="inferno")
        ax.set(
            title = "Walk: {} - Track: {} - Tor: {}".format(round(res['walk_distance'], 2), round(res['tracking_distance'], 2), round(res['torosity'], 2))
        )
        
        if not save_plots:
            plt.show()
        else:
            f.savefig(os.path.join(self.plots_save_fld, '{}.png'.format(trial_id)))
            plt.close(f)



    """
        #########################################################################################################################################################
                ANALYSIS
        #########################################################################################################################################################
    """

    def results_loader(self, name, select_bridge=None, select_escapes=None):
        res = pd.read_pickle("Processing\\trials_analysis\\torosity_results_toshelter_{}.pkl".format(name))
        res['experiment'] = [name for i in range(len(res))]

        if select_bridge is not None:
            res = res.loc[res.escape_arm == select_bridge]

        if select_escapes is not None:
            if select_escapes:
                res = res.loc[res.is_escape == "true"]
            else:
                res = res.loc[res.is_escape == "false"]

        return res

    def load_all_res(self):
        res1 = self.zscore_and_sort(self.results_loader("asymmetric", select_bridge=None, select_escapes=True))
        res2 = self.zscore_and_sort(self.results_loader("symmetric", select_bridge=None, select_escapes=True))
        res = self.zscore_and_sort(pd.concat([res1, res2], axis=0))       

        ares1 = self.zscore_and_sort(self.results_loader("asymmetric", select_bridge=None, select_escapes=None))
        ares2 = self.zscore_and_sort(self.results_loader("symmetric", select_bridge=None, select_escapes=None))
        ares = self.zscore_and_sort(pd.concat([ares1, ares2], axis=0))    

        return res, ares, res1, ares1, res2, ares2

    def inspect_results(self):
        # Get data
        res, ares, res1, ares1, res2, ares2 = self.load_all_res()

        # Focus on Torosity
        threshold = [(-1.5, -0.6), (-.025, .025), (0.8, 1.3),  (2, 10)]
        colors = ['g', 'b', 'm', 'c']

        f, axarr = plt.subplots(ncols=4, nrows=2)
        axarr = axarr.flatten()

        tot_res, esc_res = res.torosity_z, res.loc[res.is_escape == "true"].torosity_z
        _, bins, _ = axarr[0].hist(tot_res, bins=25, color='k', alpha=.55, log=True)

        _, bins, _ = axarr[1].hist(ares.torosity, bins=25, color='k', alpha=.55, log=True, label='All trials')
        axarr[1].hist(res.torosity, bins=bins, color='y', alpha=.55, log=True, label = 'escapes')

        axarr[0].legend()
        axarr[0].set(title="Torosity z-scored", ylabel="count", xlabel="z(torosity)")
        axarr[1].set(title="Torosity", ylabel="count", xlabel="torosity", xlim=[0,5])
        axarr[1].legend()

        _, bins, _, = axarr[2].hist([np.mean(t[:, 2, 0]) for t in ares.tracking_data.values], color='k', alpha=.55, label='all', bins=50)
        axarr[2].hist([np.mean(t[:, 2, 0]) for t in res.tracking_data.values], color='y', alpha=.55, label='escapes', bins=bins)
        axarr[2].set(title='Mean speed', ylabel='count', xlabel='speed')

        for th, c, ax in zip(threshold, colors, axarr[4:]):
            axarr[0].axvline(th[0], color=c)
            axarr[0].axvline(th[1], color=c)

            ax.imshow(self.agent.maze, cmap="Greys_r")

            tor = res1.loc[(res1.torosity_z <= th[1]) & (res1.torosity_z >= th[0])]
            for i,t in tor.iterrows():
                ax.scatter(t.tracking_data[:, 0, 0], t.tracking_data[:, 1, 0], c=c, alpha=.1, s=75, label="LOW tor")


            ax.set(title='Example trajectories - th: {}'.format(th), xticks=[], yticks=[])

    def plot_all_timed_toros(self):
        res, ares, res1, ares1, res2, ares2 = self.load_all_res()

        a = 1



if __name__ == "__main__":
    mazes = ["asymmetric", "symmetric"]
    for m in mazes:
        t = Torosity(m)

        # for i in range(10):
        # t.plot_one_escape_and_its_walk()

        t.analyse_all(plot=False, save_plots=True)
        # t.inspect_results()

        # t.plot_all_timed_toros()

        plt.show()









