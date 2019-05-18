import sys
sys.path.append('./')

from scipy.signal import resample

from Utilities.imports import *

from Processing.trials_analysis.all_trials_loader import Trials
from Modelling.maze_solvers.gradient_agent import GradientAgent

print("\n\n\n")

class Torosity(Trials):
    def __init__(self, mtype):
        Trials.__init__(self, exp_1_mode=True, just_escapes="true")

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
        self.results_keys = ["walk_distance", "tracking_distance", "torosity", "tracking_data", "escape_arm", "is_escape", "binned_torosity",
                            "time_out_of_t", "threat_torosity"]

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

    def threat_torosity(self, tracking, out_of_t, br):
        tracking = tracking[:out_of_t, :, :]
        return self.process_one_trial(None, br, tracking=tracking, goal=list(tracking[-1, :2, 0]))

    def time_binned_torosity(self, tracking, br):
        window_size, window_step = 10, 5

        i_start = 0
        i_end = i_start + window_size

        binned_tor_list = []
        while i_end <= tracking.shape[0]:
            binned_tracking = tracking[i_start:i_end, :, :]
            i_start += window_step
            i_end += window_step

            torosity = self.process_one_trial(None, br, tracking=binned_tracking, goal=list(binned_tracking[-1, :2, 0]))

            binned_tor_list.append((i_start, i_end, torosity))

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
            self.agent.introduce_blockage("right_large", update=True)

        # do a walk with the same geod
        walk = np.array(self.agent.walk())


        # compute stuff
        walk_distance  = np.sum(calc_distance_between_points_in_a_vector_2d(walk)) 

        # if goal is None: # ? not very elegant, but it it is to say that we check only when we are processing the whole trial
        #     if len(walk < 10): raise ValueError("Why u not walking")
        #     if walk_distance < 1: raise ValueError("Something went wrong with the walk distance")

        tracking_distance   = np.sum(calc_distance_between_points_in_a_vector_2d(tracking[:, :2, 0])) 
        torosity = tracking_distance/walk_distance

        # if (np.isnan(torosity) or np.isinf(torosity)) and goal is None: raise ValueError


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

    def analyse_all(self, plot=False, save_plots=False, bined_tor=False, threat_tor=False,):
        print("processing all trials")
        all_res = {k:[] for k in self.results_keys}

        trials = [t for i,t in self.trials.iterrows()]
        for trial in tqdm(trials):
            # Get the escape bridge
            try:
                bridge = self.bridges_lookup[trial.escape_arm]
                time_out_of_t = int(trial.time_out_of_t * trial.fps)
            except: 
                continue

            # Process whole trial
            tracking, walk, res = self.process_one_trial(trial, bridge)
            if tracking is None: continue
            
            # Binned torosity
            if bined_tor:
                res["binned_torosity"] = self.time_binned_torosity(tracking, bridge)
                # Plot binned tor
                if plot:
                    self.plot_time_binned_torosity(tracking, walk, res["binned_torosity"], res, save_plots=save_plots, trial_id=trial.trial_id)
            else:
                res["binned_torosity"] = None

            # Threat torosity
            if threat_tor:
                res['threat_torosity'] = self.threat_torosity(tracking, time_out_of_t, bridge)
            else:
                res['threat_torosity'] = None
            res['time_out_of_t'] = time_out_of_t
            
            # Put all results into main dict
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
        use_tor = "threat_torosity"

        # Get data
        res, ares, res1, ares1, res2, ares2 = self.load_all_res()
        tot_res = [x for x in res[use_tor] if not np.isnan(x)],
        esc_res =  [x for x in res.loc[res.is_escape == "true"][use_tor] if not np.isnan(x)]
        atot_res = [x for x in ares[use_tor] if not np.isnan(x)]

        # Focus on Torosity
        if use_tor == "threat_torosity":
            threshold = [(0.9, 1.0), (1.4, 1.5), (1.8, 2),  (3.5, 10)]
        else:
            threshold = [(-1.5, -0.7), (-.01, .01), (1, 1.1),  (4, 10)]
        colors = ['g', 'b', 'm', 'k']
        colormaps = ["Greens", "Blues", "Purples", "Greys"]

        # Create figure
        f, axarr = plt.subplots(ncols=4, nrows=2)
        axarr = axarr.flatten()

        # Plot escapes torosity
        _, bins, _ = axarr[0].hist(tot_res, bins=25, color='k', alpha=.55, log=True)

        # # Plot escape vs all torosity
        # _, bins, _ = axarr[1].hist(atot_res, bins=50, color='k', alpha=.55, log=True, label='All trials')
        # axarr[1].hist(res.torosity, bins=bins, color='y', alpha=.55, log=True, label = 'escapes')

        # # Plot velocity for all vs escapes
        # _, bins, _, = axarr[2].hist([np.mean(t[:, 2, 0]) for t in ares.tracking_data.values], color='k', alpha=.55, label='all', bins=15)
        # axarr[2].hist([np.mean(t[:, 2, 0]) for t in res.tracking_data.values], color='y', alpha=.55, label='escapes', bins=bins)
        # axarr[2].set(title='Mean speed', ylabel='count', xlabel='speed')

        # Correlation between overall torosity and threat torosity
        x, y_pred = linear_regression(res.torosity.values, res.threat_torosity.values)
        axarr[2].scatter(res.torosity, res.threat_torosity, c='g', s=150, alpha=.6)
        axarr[2].plot(x, y_pred, color='r', linewidth=2)

        # plot mean vel vs torosity
        mean_speeds = [np.mean(t.tracking_data[:, 2, 0]) for i,t in res.iterrows()]
        axarr[3].scatter(mean_speeds, res.threat_torosity, color='k', s=150, alpha=.6)

        # Plot RIGHT vs LEFT escape for ASYM maze as a function of threat toro
        asym_data = res.loc[res.experiment == "asymmetric"].sort_values("threat_torosity")
        axarr[1].plot(asym_data.threat_torosity, [1 if e =="right" else 0 for e in asym_data.escape_arm], color='r', linewidth=4, alpha=.3)

        # Plot examples of traces with different torosities
        for th, c, ax, cmap in zip(threshold, colors, axarr[4:], colormaps):
            # axarr[0].axvline(th[0], color=c)
            # axarr[0].axvline(th[1], color=c)

            img = np.ones_like(self.agent.maze)*100
            img[0, 0] = 0
            ax.imshow(img, cmap="Greys_r")

            tor = res.loc[(res[use_tor] <= th[1]) & (res[use_tor] >= th[0])]
            for i,t in tor.iterrows():
                if use_tor == "threat_torosity":
                    end = t.time_out_of_t
                else:
                    end = -1

                x,y = t.tracking_data[:end, 0, 0], t.tracking_data[:end, 1, 0]
                ax.plot(x,y,  linestyle="-", linewidth=4, color=c, alpha=0.5,label="LOW tor")
                # ax.scatter(x,y, c=np.arange(len(x)), cmap=cmap, vmin=10,  alpha=.9, s=50)

                


            if use_tor == "threat_torosity": 
                ax.set(title='Example trajectories - th: {}'.format(th), xticks=[], yticks=[], xlim=[100, 150], ylim=[225, 150])
            else:
                ax.set(title='Example trajectories - th: {}'.format(th), xticks=[], yticks=[])

        # Set axes
        axarr[0].   legend()
        axarr[0].set(title="{} z-scored".format(use_tor), ylabel="count", xlabel="z(torosity)")
        axarr[1].set(title="Escape arm vs Torosity", xlabel="threat_torosity", yticks=[0,1], yticklabels=["left", "right"])
        axarr[1].legend()
        axarr[2].set(title='Total vs Threat Torosity', xlabel='overall', ylabel='threat')
        axarr[3].set(xlabel='mean speed', ylabel='torosity')



    def plot_all_timed_toros(self):
        res, ares, res1, ares1, res2, ares2 = self.load_all_res()

        f, ax = plt.subplots()

        for i, row in res.iterrows():
            ax.plot(line_smoother([r for i,e,r in row.binned_torosity]))



if __name__ == "__main__":
    mazes = ["symmetric"]
    for m in mazes:
        t = Torosity(m)

        # t.analyse_all(bined_tor=False, threat_tor=True, plot=False, save_plots=True)

        t.inspect_results()



        plt.show()









