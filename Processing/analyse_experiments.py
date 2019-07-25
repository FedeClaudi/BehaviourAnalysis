# %%
import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

# if __name__ == "__main__": # avoid re importing the tables for every core in during bayesian modeeling
from Utilities.imports import *


from Modelling.bayesian.bayes_V3 import Bayes
from Modelling.maze_solvers.gradient_agent import GradientAgent

# %matplotlib inline
import pickle
import pymc3 as pm

# %%
# Define class
class ExperimentsAnalyser(Bayes):
    maze_designs = {0:"three_arms", 1:"asymmetric_long", 2:"asymmetric_mediumlong", 3:"asymmetric_mediumshort", 4:"symmetric", -1:"nan"}
    naive_lookup = {0: "experienced", 1:"naive", -1:"nan"}
    lights_lookup = {0: "off", 1:"on", 2:"on_trials", 3:"on_exploration", -1:"nan"}

    colors = {0:blue, 1:red, 2:green, 3:magenta, 4:orange, -1:white}
    arms_colors = {"Left_Far":green, "Left_Medium":green, "Right_Medium":red, "Right_Far":red, "Centre":magenta}

    # ! important 
    max_duration_th = 8.0

    # Folders
    if sys.platform != "darwin":
        metadata_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\Psychometric"
    else:
        metadata_folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/Psychometric"

    def __init__(self):
        Bayes.__init__(self)
        
        self.conditions = dict(
                    maze1 =  self.get_sessions_trials(maze_design=1, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze2 =  self.get_sessions_trials(maze_design=2, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze3 =  self.get_sessions_trials(maze_design=3, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze4 =  self.get_sessions_trials(maze_design=4, naive=None, lights=None, escapes=None, escapes_dur=True),
                )

        self.session_metadata = pd.DataFrame((Session * Session.Metadata - "maze_type=-1"))

    # def __str__(self):
    #     def get_summary(df):
    #         summary = dict(maze=[], tot_mice=[], naive=[])
    #         for maze_id, maze_name in self.maze_designs.items():
    #             if maze_id == -1: continue
                    
    #             maze_data = df.loc[df.maze_type == maze_id]

    #             summary["maze"].append(maze_name)
    #             summary["tot_mice"].append(len(maze_data))
    #             summary["naive"].append(len(maze_data.loc[maze_data.naive == 1]))

    #         summary = pd.DataFrame(summary)
    #         return summary

    #     data = self.session_metadata
    #     summary = get_summary(data)
    #     print("Sessions per experiment\n", summary)
    #     print()
    #     return ""

    # def __repr__(self): 
    #     self.__str__()
    #     return ""

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         GET DATA
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """

    def get_sessions_by_condition(self, maze_design=None, naive=None, lights=None, df=False):
        data = Session * Session.Metadata  - 'experiment_name="Foraging"'  - "maze_type=-1"
        if maze_design is not None:
            data = (data & "maze_type={}".format(maze_design))
        if naive is not None:
            data = (data & "naive={}".format(naive))
        if lights is not None:
            data = (data & "lights={}".format(lights))

        if df:
            return pd.DataFrame((data).fetch())
        else: return data

    def get_sessions_tracking(self, bp="body",  maze_design=None, naive=None, lights=None):
        data = self.get_sessions_by_condition(maze_design, naive, lights, df=False)
        andtracking = (data * TrackingData.BodyPartData & "bpname='{}'".format(bp))
        return pd.DataFrame(andtracking.fetch())

    def get_sessions_trials(self, maze_design=None, naive=None, lights=None, escapes=False, escapes_dur=True):
        # Given a dj query with the relevant sessions, fetches the corresponding trials from AllTrials
        sessions = self.get_sessions_by_condition(maze_design, naive, lights, df=True)
        ss = set(sorted(sessions.uid.values))

        all_trials = pd.DataFrame(AllTrials.fetch())
        if escapes:
            all_trials = all_trials.loc[all_trials.is_escape == "true"]

        if escapes_dur:
            all_trials = all_trials.loc[all_trials.escape_duration <= self.max_duration_th]
        trials = all_trials.loc[all_trials.session_uid.isin(ss)]

        return trials

    def get_binary_trials_per_condition(self, conditions):
        # ? conditions should be a dict whose keys should be a list of strings with the names of the different conditions to be modelled
        # ? the values of conditions should be a a list of dataframes, each specifying the trials for one condition (e.g. maze design) and the session they belong to

        # Parse data
        # Get trials
        trials = {c:[] for c in conditions.keys()}
        for condition, df in conditions.items():
            sessions = sorted(set(df.session_uid.values))
            for sess in sessions:
                trials[condition].append([1 if "right" in arm.lower() else 0 for arm in df.loc[df.session_uid==sess].escape_arm.values])

        # Get hits and number of trials
        hits = {c:[np.sum(t2) for t2 in t] for c, t in trials.items()}
        ntrials = {c:[len(t2) for t2 in t] for c,t in trials.items()}
        p_r = {c: [h/n for h,n in zip(hits[c], ntrials[c])] for c in hits.keys()}
        n_mice = {c:len(v) for c,v in hits.items()}
        return hits, ntrials, p_r, n_mice

    def save_trials_to_pickle(self):
        for k, df in self.conditions.items():
            save_df(df, os.path.join(self.metadata_folder, k+".pkl"))

    def load_trials_from_pickle(self):
        names = ["maze1", "maze2", "maze3", "maze4"]
        return {n:load_df(os.path.join(self.metadata_folder, n+".pkl")) for n in names}

    def get_arms_lengths_with_agent(self, load=False):
        if not load:
            # Get Gradiend Agent and maze arms images
            agent = GradientAgent(grid_size=1000, start_location=[515, 208], goal_location=[515, 720])

            if sys.platform == "darwin":
                maze_arms_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/maze_solvers/good_single_arms"
            else:
                maze_arms_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\maze_solvers\\good_single_arms"
            
            arms = [os.path.join(maze_arms_fld, a) for a in os.listdir(maze_arms_fld) if "jpg" in a or "png" in a]
            arms_data = dict(maze=[], n_steps=[], torosity=[], distance=[])
            # Loop over each arm
            for arm in arms:
                if "centre" in arm: continue
                print("getting geo distance for arm: ",arm)
                # ? get maze, geodesic and walk
                agent.maze, agent.free_states = agent.get_maze_from_image(model_path=arm)
                agent.geodesic_distance = agent.geodist(agent.maze, agent.goal_location)
                walk = agent.walk()
                agent.plot_walk(walk)


                # Process walks to get lengths and so on
                arms_data["maze"].append(os.path.split(arm)[1].split(".")[0])
                arms_data["n_steps"].append(len(walk))
                arms_data["distance"].append(round(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk)))))
                threat_shelter_dist = calc_distance_between_points_2d(agent.start_location, agent.goal_location)
                arms_data["torosity"].append(np.sum(calc_distance_between_points_in_a_vector_2d(np.array(walk))) / threat_shelter_dist)
            
            arms_data = pd.DataFrame(arms_data)
            print(arms_data)
            arms_data.to_pickle(os.path.join(self.metadata_folder, "geoagent_paths.pkl"))
            plt.show()
            return arms_data
        else:
            return pd.read_pickle(os.path.join(self.metadata_folder, "geoagent_paths.pkl"))

    def merge_conditions_trials(self, dfs):
        # dfs is a list of dataframes with the trials for each condition

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.append(df)
        return merged

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         ANALYSE STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    def bayes_by_condition_analytical(self, load=True, mode="grouped", plot=True):
        if not load: raise NotImplementedError
        else:
            data = self.load_trials_from_pickle()

        modes, means = self.analytical_bayes_individuals(conditions=None, data=data, mode=mode, plot=plot)
        return modes, means

    def bayes_by_condition(self, conditions=None,  load=False, tracefile="a.pkl", plot=True):
        tracename = os.path.join(self.metadata_folder, tracefile)

        if conditions is None:
            conditions = self.conditions

        if not load:
            trace = self.model_hierarchical_bayes(conditions)
            self.save_bayes_trace(trace, tracename)
            trace = pm.trace_to_dataframe(trace)
        else:
            trace = self.load_trace(tracename)

        # Plot by condition
        good_columns = {c:[col for col in trace.columns if col[0:len(c)] == c] for c in conditions.keys()}
        condition_traces = {c:trace[cols].values for c, cols in good_columns.items()}

        if plot:
            f, axarr = plt.subplots(nrows=len(conditions.keys()))
            for (condition, data), color, ax in zip(condition_traces.items(), ["w", "m", "g", "r", "b", "orange"], axarr):
                for i in np.arange(data.shape[1]):
                    if i == 0: label = condition
                    else: label=None
                    sns.kdeplot(data[:, i], ax=ax, color=color, shade=True, alpha=.15)

                ax.set(title="p(R) posteriors {}".format(condition), xlabel="p(R)", ylabel="pdf", facecolor=[.2, .2, .2])
                ax.legend()            
            plt.show()

        return condition_traces

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         ANALYSE STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    def escape_definition_investigation(self):
        conditions = dict(
                    maze1 =  self.get_sessions_trials(maze_design=1, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze2 =  self.get_sessions_trials(maze_design=2, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze3 =  self.get_sessions_trials(maze_design=3, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze4 =  self.get_sessions_trials(maze_design=4, naive=None, lights=None, escapes=None, escapes_dur=True),
                )

        f,ax = plt.subplots()
        same, same_detailed = [], {"yy":0, "yn":0, "ny":0, "nn":0}
        for condition, df in conditions.items():
            sess_same = []
            for i, trial in df.iterrows():
                if trial.is_escape == "true":
                    if trial.escape_duration <= 10: 
                        sess_same.append(1)
                        same_detailed["yy"] += 1
                    else: 
                        sess_same.append(0)
                        same_detailed["yn"] += 1
                else:
                    if trial.escape_duration <= 10: 
                        sess_same.append(0)
                        same_detailed["ny"] += 1
                    else: 
                        sess_same.append(1)    
                        same_detailed["nn"] += 1
            same.extend(sess_same)

            ax.hist(df.loc[df.escape_arm == "Right_Medium"].escape_duration, color=red, alpha=.5,  bins=20, density=False)
            ax.hist(df.loc[df.escape_arm == "Left_Medium"].escape_duration, color=green, alpha=.5, bins=20, density=False)
            ax.hist(df.loc[df.escape_arm == "Left_Far"].escape_duration, color=green, alpha=.5, bins=20, density=False)

        ax.axvline(self.max_duration_th, color="m", lw=3)
        make_legend(ax)
        print("{} of trials have matching definition".format(np.mean(same)))
        print(same_detailed)


    def escape_thershold_effect(self):
        f, axarr = plt.subplots(nrows=5)

        ths = np.linspace(0, 20, 41)

        for th in tqdm(ths):
            self.max_duration_th = th
            conditions = dict(
                    maze1 =  self.get_sessions_trials(maze_design=1, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze2 =  self.get_sessions_trials(maze_design=2, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze3 =  self.get_sessions_trials(maze_design=3, naive=None, lights=None, escapes=None, escapes_dur=True),
                    maze4 =  self.get_sessions_trials(maze_design=4, naive=None, lights=None, escapes=None, escapes_dur=True),
                )

            hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(conditions)
            for i, (condition, pr) in enumerate(p_r.items()):
                axarr[i].scatter([th for _ in np.arange(len(pr))], pr, alpha=.7, color=self.colors[i+1], s=100)
                # axarr[i].errorbar(i, np.mean(pr), yerr=np.std(pr),  **white_errorbar)

            self.save_trials_to_pickle()
            modes, means = self.bayes_by_condition_analytical(load=True, plot=False)
            for i, (maze, m) in enumerate(means.items()):
                axarr[4].scatter(th, m, color=self.colors[i+1], s=200)

        titles = list(conditions.keys()) + ["grouped bayes means"]
        for ax, ttl in zip(axarr, titles):
            ax.set(title=ttl, ylim=[-0.1, 1.1])

    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         PLOT STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    def plot_trials_tracking(self):
        trials = self.get_sesions_trials(escapes=True)
        self.plot_tracking(trials, origin=False)

    def plot_sessions_tracking(self):
        tracking = self.get_sessions_tracking()
        self.plot_tracking(tracking, maxT=10000)

    def plot_tracking(self, tracking, ax=None, colorby=None, color="w", origin=False, minT=0, maxT=-1):
        if ax is None: f, ax = plt.subplots()

        for i, trial in tracking.iterrows():
            if not origin: tr = trial.tracking_data
            else: tr = trial.outward_tracking_data

            if colorby == "arm": 
                kwargs = {"color":self.arms_colors[trial.escape_arm]}
            elif colorby == "speed": 
                kwargs = {"c":tr[minT:maxT, 2], "cmap":"gray"}
            else: 
                kwargs = {"color":color}
                
            ax.scatter(tr[minT:maxT, 0], tr[minT:maxT, 1], alpha=.25, s=10, **kwargs) 
        ax.set(facecolor=[.05, .05, .05])

    def tracking_custom_plot(self):
        f, axarr = plt.subplots(ncols=3)
        
        for i in np.arange(4):
            mazen = i + 1
            tracking = self.get_sessions_trials(maze_design=mazen, lights=1, escapes=True)
            self.plot_tracking(tracking, ax=axarr[0], colorby=None, color=self.colors[mazen])
            self.plot_tracking(tracking, ax=axarr[1], colorby="speed", color=self.colors[mazen])
            self.plot_tracking(tracking, ax=axarr[2], colorby="arm", color=self.colors[mazen])

        for ax in axarr:
            ax.set(xlim=[100, 720], ylim=[100, 720])

    def plot_pr_by_condition(self):
        conditions = self.conditions

        hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(conditions)

        f, ax = plt.subplots()

        for i, (condition, pr) in enumerate(p_r.items()):
            ax.scatter(np.random.normal(i, .1, size=len(pr)), pr, alpha=.7, color=self.colors[i+1], s=250, label=condition)
            ax.scatter(i, np.mean(pr), alpha=1, color="w", s=500, label=None)
            ax.errorbar(i, np.mean(pr), yerr=np.std(pr),  **white_errorbar)

        ax.set(facecolor=[.2, .2, .2], ylim=[-0.05, 1.05],ylabel="p(R)", title="p(R) per mouse per maze",
                 xticks = np.arange(len(conditions.keys())), xticklabels = conditions.keys())
        ax.legend()

    def plot_pr_vs_time(self):
        # loop over different settings for kde
        for bw in [60]:
            # crate figure
            f, axarr = create_figure(subplots=True, nrows=5, ncols=2, sharex=True)
            leftcol, rightcol = [0, 2, 4, 6, 8], [1, 3, 5, 7, 9]

            # loop over experiments
            for i, (condition, df) in enumerate(self.conditions.items()):
                ax = axarr[leftcol[i]]
                axspeed = axarr[rightcol[i]]
                times, ones, zeros, speeds = [], [], [], []
                # loop over trials
                for n, (_, trial) in enumerate(df.iterrows()):
                    # get time of trial and escape arm
                    x = trial.stim_frame_session / trial.fps
                    if 'Right' in trial.escape_arm:
                        y = 1
                        ones.append(x)
                    else:
                        y = 0
                        zeros.append(x)

                    # Get escape speed
                    escape_speed = np.mean(trial.tracking_data[:, 2]) / trial.fps
                    times.append(x)
                    speeds.append(escape_speed)

                    # plot
                    ax.scatter(x, y, color=self.colors[i+1], s=50, alpha=.5)

                # linear regression on speed and
                sns.regplot(times, speeds, ax=axspeed, scatter=True, order=1, scatter_kws=dict(s=75, color=desaturate_color(self.colors[i+1], k=.5)),
                                line_kws=dict(color=self.colors[i+1], lw=2, alpha=1), truncate=True,)

                # Plot KDEs
                ax, kde_right = plot_kde(ax, fit_kde(ones, bw=bw), .8, invert=True, normto=.25, color=self.colors[i+1])
                ax, kde_left = plot_kde(ax, fit_kde(zeros, bw=bw), .2, invert=False, normto=.25, color=self.colors[i+1])

                # Plot ratio of KDEs in last plot
                xxx = np.linspace(np.max([np.min(kde_right.support), np.min(kde_left.support)]), np.min([np.max(kde_right.support), np.max(kde_left.support)]), 1000)
                ratio = [kde_right.evaluate(xx)/(kde_right.evaluate(xx)+kde_left.evaluate(xx)) for xx in xxx]
                axarr[leftcol[4]].plot(xxx, ratio, lw=3, color=self.colors[i+1], label=condition)

            # Set axes correctly
            for i, (ax, maze) in enumerate(zip(axarr, list(self.conditions.keys()))):
                if i in leftcol:
                    kwargs = dict( ylim=[-.1, 1.1], yticklabels=["left", "right"], ylabel="escape", yticks=[0, 1],  ) 
                else:
                    kwargs = dict(ylabel="90th perc of escape speed (a.u)")

                ax.set(title=maze, xlabel="time (min)",  xticks=[x*60 for x in np.linspace(0, 100, 11)], xticklabels=np.linspace(0, 100, 11), **kwargs)

            ortholines(axarr[4], [0,], [.5])
            axarr[leftcol[4]].set(title="balance over time", xlabel="time (min)", ylabel="R / (R+L)")
            make_legend(axarr[4])
            f.tight_layout()


if __name__ == "__main__":
    ea = ExperimentsAnalyser()
    # print(ea)
    ea.save_trials_to_pickle()
    # ea.tracking_custom_plot()
    # ea.plot_pr_by_condition()

    # ea.escape_definition_investigation()
    # ea.escape_thershold_effect()

    # ea.plot_pr_vs_time()

    

    # ea.bayes_by_condition(conditions=None,  load=False, tracefile="psychometric_individual_bayes.pkl", plot=False)

    plt.show()


