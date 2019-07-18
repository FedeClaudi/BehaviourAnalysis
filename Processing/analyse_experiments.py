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
    arms_colors = {"Left_Far":green, "Left_Medium":green, "Right_Medium":green, "Right_Far":green, "Centre":magenta}

    # Folders
    if sys.platform != "darwin":
        metadata_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\Psychometric"
    else:
        metadata_folder = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/Psychometric"

    def __init__(self):
        Bayes.__init__(self)

        # self.session_metadata = pd.DataFrame((Session * Session.Metadata - "maze_type=-1"))

    # def __str__(self):
    #     def get_summary(df, lights=1):
    #         summary = dict(maze=[], tot_mice=[], naive=[], n_stimuli=[], n_escapes=[])
    #         for maze_id, maze_name in self.maze_designs.items():
    #             if maze_id == -1: continue
                    
    #             maze_data = df.loc[df.maze_type == maze_id]

    #             summary["maze"].append(maze_name)
    #             summary["tot_mice"].append(len(maze_data))
    #             summary["naive"].append(len(maze_data.loc[maze_data.naive == 1]))
    #             summary["n_stimuli"].append(len(self.get_sesions_trials(maze_design=maze_id, naive=None, lights=lights, escapes=False)))
    #             summary["n_escapes"].append(len(self.get_sesions_trials(maze_design=maze_id, naive=None, lights=lights, escapes=True)))

    #         summary = pd.DataFrame(summary)
    #         return summary

    #     data = self.session_metadata
    #     # Get how many mice were done with the lights on on each maze, divide by naive and not naive
    #     lights_on_data = data.loc[data.lights==1]
    #     summary = get_summary(lights_on_data, lights=1)

    #     print("Sessions per experiment - lights ON")
    #     print(summary)


    #     lights_off_data = data.loc[data.lights==0]
    #     summary = get_summary(lights_off_data, lights=0)

    #     print("\n\Sessions per experiment - lights OFF")
    #     print(summary)
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

    def get_sesions_trials(self, maze_design=None, naive=None, lights=None, escapes=True):
        # Given a dj query with the relevant sessions, fetches the corresponding trials from AllTrials
        sessions = self.get_sessions_by_condition(maze_design, naive, lights, df=True)
        ss = set(sorted(sessions.uid.values))

        all_trials = pd.DataFrame(AllTrials.fetch())
        if escapes:
            all_trials = all_trials.loc[all_trials.is_escape == "true"]
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
        conditions = dict(
            maze1 =  self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
            maze2 =  self.get_sesions_trials(maze_design=2, naive=None, lights=1, escapes=True),
            maze3 =  self.get_sesions_trials(maze_design=3, naive=None, lights=1, escapes=True),
            maze4 =  self.get_sesions_trials(maze_design=4, naive=None, lights=1, escapes=True),
        )

        for k, df in conditions.items():
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



    """
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
                         ANALYSE STUFF
    ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    """
    def bayes_by_condition_analytical(self, load=True, mode="grouped", plot=True):
        if not load: raise NotImplementedError
        else:
            data = self.load_trials_from_pickle()

        modes = self.analytical_bayes_individuals(conditions=None, data=data, mode=mode, plot=plot)
        return modes

    def bayes_by_condition(self, conditions=None,  load=False, tracefile="a.pkl", plot=True):
        tracename = os.path.join(self.metadata_folder, tracefile)

        if conditions is None:
            conditions = dict(
                    maze1 =     self.get_sesions_trials(maze_design=1, naive=None, lights=1, escapes=True),
                    maze2 =     self.get_sesions_trials(maze_design=2, naive=None, lights=1, escapes=True),
                    maze3 =     self.get_sesions_trials(maze_design=3, naive=None, lights=1, escapes=True),
                    maze4 =     self.get_sesions_trials(maze_design=4, naive=None, lights=1, escapes=True),
                )

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
                c = self.arms_colors[trial.escape_arm]
                cmap = None
            elif colorby == "speed": 
                c=tr[minT:maxT, 2]
                cmap = "gray"
            else: 
                c = color
                cmap = None
                
            ax.scatter(tr[minT:maxT, 0], tr[minT:maxT, 1], c=c,  cmap=cmap, alpha=.25, s=10) # cmap="gray", vmin=0, vmax=10,
        ax.set(facecolor=[.05, .05, .05])

    def tracking_custom_plot(self):
        # f, axarr = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
        # axarr = axarr.flatten()
        f, axarr = plt.subplots(ncols=3)
        
        # for i, ax in enumerate(axarr):
        for i in np.arange(4):
            mazen = i + 1
            tracking = self.get_sesions_trials(maze_design=mazen, lights=1, escapes=True)
            self.plot_tracking(tracking, ax=axarr[0], colorby=None, color=self.colors[mazen])
            self.plot_tracking(tracking, ax=axarr[1], colorby="speed", color=self.colors[mazen])
            self.plot_tracking(tracking, ax=axarr[2], colorby="arm", color=self.colors[mazen])

        for ax in axarr:
            ax.set(xlim=[100, 720], ylim=[100, 720])

    def tracking_aligned_to_start_plot(self):
        # TODO
        pass

    def plot_pr_by_condition(self):
        conditions = dict(
            maze1 =  self.get_sesions_trials(maze_design=1, naive=0, lights=1, escapes=True),
            maze2 =  self.get_sesions_trials(maze_design=2, naive=0, lights=1, escapes=True),
            maze3 =  self.get_sesions_trials(maze_design=3, naive=0, lights=1, escapes=True),
            maze4 =  self.get_sesions_trials(maze_design=4, naive=0, lights=1, escapes=True),
        )

        hits, ntrials, p_r, n_trials = self.get_binary_trials_per_condition(conditions)

        f, ax = plt.subplots()

        for i, (condition, pr) in enumerate(p_r.items()):
            ax.scatter(np.random.normal(i, .1, size=len(pr)), pr, alpha=.7, color=self.colors[i+1], s=250, label=condition)
            ax.scatter(i, np.mean(pr), alpha=1, color="w", s=500, label=None)

        ax.set(facecolor=[.2, .2, .2], ylim=[-0.05, 1.05],ylabel="p(R)", title="p(R) per mouse per maze",
                 xticks = np.arange(len(conditions.keys())), xticklabels = conditions.keys())
        ax.legend()

        

if __name__ == "__main__":
    ea = ExperimentsAnalyser()
    # print(ea)
    ea.bayes_by_condition()
    # ea.save_trials_to_pickle()
    # ea.tracking_custom_plot()
    # ea.plot_pr_by_condition()
    plt.show()



#%%