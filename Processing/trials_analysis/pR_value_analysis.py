]import sys
sys.path.append("./")

from Utilities.imports import *

from itertools import product

from Modelling.maze_solvers.gradient_agent import GradientAgent as Agent
from Processing.plot.poster_plotter import Plotter as analyser_trials
"""
    Try to get a quantitative way to explain why in the ASYM maze the p(R) = .84

"""





class Analyzer:
    xx = np.linspace(0.01, 0.99, 1000)
    opt_p = namedtuple("optp", "length rLen iLen uLen theta_start theta_tot")

    gamma = 1  # weight of differences in angles
    sigma = 1  # weight of differences in length

    def __init__(self, load=False):
        if not load:
            self.params = {}
            self.get_sym_params()

            self.get_mbv3_params()
            self.get_asym_params()
            self.get_mbv2_params()
            save_yaml("Processing\\trials_analysis\\params.yml", self.params)
        else: 
            params = load_yaml("Processing\\trials_analysis\\params.yml")

            self.params = {k:{k2:self.opt_p(*v2) for k2,v2 in v.items()} for k,v in params.items()}

        self.load_asym_bayes_posteriors()
        self.get_mbv2_probs()


    def load_asym_bayes_posteriors(self):
        # Get the asym posteriors
        f = "Modelling/bayesian/hierarchical_v2.pkl"
        data = pd.read_pickle(f)
        cols_to_drop = [c for c in data.columns if "asym_prior" not in c]
        data = data.drop(cols_to_drop, axis=1)

        # Get the modes
        self.asym_pR_individuals = data.mean().values

    def get_mbv2_probs(self):
        # trials_data = "Processing\\trials_analysis\\data\\mbv2_trials.yml"
        # data = load_yaml(trials_data)
        # sessions = list(data.keys())

        # analyser = analyser_trials()
        # arms = ["B0", "A0", "A1",  "B1"]
        # escapes = {a:0 for a in arms}

        # # Get all the speeds to get a threshold for escapes
        # speeds = []
        # for sess_name in sessions:
        #     # Get the tracking data corresponding to it
        #     tr = (TrackingData.BodyPartData & "bpname='body'" & "session_name='{}'".format(sess_name)).fetch1("tracking_data")
        #     speeds.append(tr[:, 2])
        # speed_th = np.percentile(np.hstack(speeds), 50)
        # a = 1


        # for sess_name in sessions:
        #     # Get the tracking data corresponding to it
        #     tr = (TrackingData.BodyPartData & "bpname='body'" & "session_name='{}'".format(sess_name)).fetch1("tracking_data")

        #     # Get the stimuli correspodnign to session
        #     stims = (Stimuli & "session_name='{}'".format(sess_name)).fetch()

        #     # Get hadn data for session
        #     arms, maze_states = data[sess_name][:2]

        #     # for each stim get the outcome
        #     for arm, state, trial in zip(arms, maze_states, stims):
        #         if arm == "n": continue
        #         # Get when the mouse reached the shelter
        #         start, end = trial['overview_frame'], int(trial['overview_frame'] + trial['duration']*analyser.fps)
        #         try:
        #             end = np.where((tr[start:, 0] > 440) & 
        #                 (tr[start:, 0] < 560) & 
        #                 (tr[start:, 1] > 775) & 
        #                 (tr[start:, 1] < 910) )[0][0] + start
        #         except: continue
        #         if end - start > 20*analyser.fps: end = start + 20*analyser.fps

        #         carm, mean_speed = analyser.look_at_trial_in_detail(tr[start:end])
        #         if carm is None: continue

        #         mean_speed = np.mean(tr[start:end, 2])

        #         if mean_speed < speed_th: 
        #             print("too slow")
        #             continue
        #         else:
        #             print("fast")
        #             escapes[arm] += 1

        #         if state != 0: break

        # tot_trials = np.sum(list(escapes.values()))
        # escapes_probs = {a:k/tot_trials for a,k in escapes.items()}

        # self.mbv2_pS = escapes_probs['A0'] + escapes_probs['A1']
        self.mbv2_pS = 0.82
        a = 1






        # arms = ["B0", "A0", "A1",  "B1"]
        # escapes = {a:0 for a in arms}

        # for session in sessions:
        #     if len(data[session][0]) != len(data[session][1]): raise ValueError(session, len(data[session][0]), len(data[session][1]))
        #     if session not in data.keys(): continue
        #     arms, maze_states = data[session][:2]
        #     for arm, maze_state in zip(arms, maze_states):
        #         if arm.lower() == "n": continue

        #         if maze_state != 0: 
        #             escapes[arm] += 1  # still count it
        #             break
        #         escapes[arm] += 1

        # tot_trials = np.sum(list(escapes.values()))
        # escapes_probs = {a:k/tot_trials for a,k in escapes.items()}

        # self.mbv2_pS = escapes_probs['A0'] + escapes_probs['A1']

    @staticmethod
    def calc_p_given_n(t1, t2, l1, l2, n):
        num = (1/t1 + l1)**n
        den = num + (1/t2 + l2)**n
        return round(num/den, 2)

    
    def calc_n_given_p(self, t1, t2, l1, l2, p):
        A = self.gamma*(np.radians(360) - t1) + l1/self.sigma
        B = self.gamma*(np.radians(360) - t2) + l2/self.sigma
        ln = math.log
        if A != B:
            n = ln(p/(1-p)) / ln(A/B)
            return round(n, 2)
        else: 
            return .5
        

    @staticmethod
    def process_options(options, key):
        options_len = {k:len(v) for k,v in options.items()}
        rL = {k:v/options_len[key] for k,v in options_len.items()}
        iL = {k:1-v for k,v in rL.items()}
        uL = {k:1/v for k,v in rL.items()}
        return options_len, rL, iL, uL

    def get_asym_params(self):
        agent = Agent()
        options_len, rL, iL, uL = self.process_options(agent.options, "left_large")

        self.params['asym'] = dict(
            left = self.opt_p(options_len['left_large'], rL['left_large'], iL['left_large'], uL['left_large'], 45, 180),
            right = self.opt_p(options_len['right_large'], rL['right_large'], iL['right_large'], uL['right_large'], 45, 135)
        )

    def get_sym_params(self):
        agent = Agent(maze_design="Square Maze.png", maze_type="symmetric", start_location=(19, 28))
        options_len, rL, iL, uL = self.process_options(agent.get_maze_options(), "left")

        self.params['sym'] = dict(
            left = self.opt_p(options_len['left'], rL['left'], iL['left'], uL['left'], 45, 135),
            right = self.opt_p(options_len['right'], rL['right'], iL['right'], uL['right'], 45, 135)
        )


    def get_mbv2_params(self):
        agent = Agent(maze_type = "modelbased", maze_design="ModelBased2.png")
        options_len, rL, iL, uL = self.process_options(agent.get_maze_options(plot=False), "beta0")

        self.params['mbv2'] = dict(
            short = self.opt_p(options_len['alpha0'], rL['alpha0'], iL['alpha0'], uL['alpha0'], 45, 180),
            long = self.opt_p(options_len['beta0'], rL['beta0'], iL['beta0'], uL['beta0'], 90, 270)
        )


    def get_mbv3_params(self):
        agent = Agent(maze_type = "modelbased", maze_design="MBV3.png")
        options_len, rL, iL, uL = self.process_options(agent.get_maze_options(plot=False), "beta1")

        self.params['mbv3'] = dict(
            short = self.opt_p(options_len['alpha0'], rL['alpha0'], iL['alpha0'], uL['alpha0'], 45, 180),
            long = self.opt_p(options_len['beta1'], rL['beta1'], iL['beta1'], uL['beta1'], 90, 270)
        )

    def get_n_line(self, exp, arms):
        n_line = [self.calc_n_given_p(np.radians(self.params[exp][arms[0]].theta_tot),
                                            np.radians(self.params[exp][arms[1]].theta_tot), 
                                            self.params[exp][arms[0]].iLen, 
                                            self.params[exp][arms[1]].iLen,
                                            p) for p in self.xx]
        return n_line

    def summary_plot(self):
        f, ax = plt.subplots()

        # Get the curve for each experiment
        asym_n_line = self.get_n_line("asym", ["right", "left"])
        mbv2_n_line = self.get_n_line("mbv2", ["short", "long"])
        mbv3_n_line = self.get_n_line("mbv3", ["short", "long"])

        sym_n_line = [.5 for x in self.xx]

        ax.plot(asym_n_line, self.xx, c='b', label="asym N line")
        ax.plot(mbv2_n_line, self.xx, c='g', label="MBV2 N line")
        ax.plot(mbv3_n_line, self.xx, c='r', label="MBV3 N line")
        ax.axhline(0.5, color='k', label="sym N line")

        # Get each mouse's params in ASYM
        nn = [self.calc_n_given_p(np.radians(self.params["asym"]["right"].theta_tot),
                                    np.radians(self.params["asym"]["left"].theta_tot), 
                                    self.params["asym"]["right"].iLen, 
                                    self.params["asym"]["left"].iLen,
                                    p) for p in self.asym_pR_individuals]
        for ni, p in zip(nn, self.asym_pR_individuals):
            ax.scatter(ni, p, color="b", alpha=.5, s=100)
        ax.axvline(np.mean(nn), color="b", alpha=.3)

        # Get expected N given p(short) in MBV2
        theoretical_n = self.calc_n_given_p(np.radians(self.params["mbv2"]["short"].theta_tot),
                                    np.radians(self.params["mbv2"]["long"].theta_tot), 
                                    self.params["mbv2"]["short"].iLen, 
                                    self.params["mbv2"]["long"].iLen,
                                    self.mbv2_pS)
        ax.axvline(theoretical_n, color="g", alpha=.3)

        # ax.scatter(mean_nn, mean_pR, color="r", s=500)
        # ax.scatter(mean_nn, theoretical_p, color="m", s=500)

        ax.legend()
        ax.set(xlabel="N", ylabel="pR", title="$pR = A^n / (A^n + B^n)$ - gamma:{}".format(self.gamma))
        plt.show()

        a = 1

 

if __name__ == "__main__":
    a = Analyzer()
    a.summary_plot()

    plt.show()


