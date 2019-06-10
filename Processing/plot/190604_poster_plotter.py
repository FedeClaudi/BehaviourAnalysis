import sys
sys.path.append('./')  
from Utilities.imports import *
from database.database_fetch import get_maze_template

from Processing.trials_analysis.TrialsMainClass import TrialsLoader

class Plotter(TrialsLoader):
    fps = 40
    asym_color = "#9575cd"
    sym_color = "#689f38"

    asym_experiments = ["PathInt2", "PathInt2 - L"]
    sym_experiments = ["Square Maze"]

    asym_template = get_maze_template(exp = "PathInt2")
    sym_template = get_maze_template(exp = "Square Maze")
    sym_template = get_maze_template(exp = "Square Maze")
    mb_template = np.rot90(get_maze_template(exp = "Model Based V2"),2)

    good_mb_uid = 286
    good_mb_session_name = "190425_CA602"
    good_mb_exclude_trials = [0, 2, 14, 19]
    good_mb_closed = 4

    mb_pre_col = "#9575cd"
    mb_post_col = "#ec407a"


    def __init__(self):
        TrialsLoader.__init__(self)
        
        self.data = (self.data & "bpname='body'")

        self.mb_trials_metadata = load_yaml("Processing/trials_analysis/data/mbv2_trials.yml")

    @staticmethod
    def look_at_trial_in_detail(tracking):
        # Look at velocity on threat
        try: 
            out_of_t = np.where(tracking[:, -1] != 1)[0][0]
        except:
            return None, None
        mean_speed = np.nanmean(tracking[:out_of_t, 2])



        # Look at outcomes devided by arm
        try: 
            on_arm = np.where(tracking[:, 1] >= 400)[0][0]
        except: return None, None
        else:
            x_on_arm = tracking[on_arm, 0]
            if x_on_arm < 285: arm = 0
            elif x_on_arm < 500: arm = 1
            elif x_on_arm < 725: arm = 2
            else: arm = 3

            return arm, mean_speed


    def plot_escapes_mbv2(self):
        f, axarr = plt.subplots(ncols=4)

        for ax in axarr:
            ax.imshow(self.mb_template)

        self.data = (self.data & "experiment_name='Model Based V2'" & "stim_type='audio'" & "duration='9.0'")

        cc = ["r", "g", "b", "c"]

        prec, postc = 0, 0
        for i, trial in enumerate(self.data):
            try:
                metadata = self.mb_trials_metadata[trial['session_name']][1]
            except: continue
            if 1 in metadata:
                first_close = metadata.index(1)
            else: first_close = 10000

            trial_n  = int(trial['stimulus_uid'].split("_")[-1])

            if trial_n < first_close:
                ax = axarr[0]
                col = self.mb_pre_col
                prec += 1
            elif trial_n == first_close:
                ax = axarr[1]
                col = 'r'
            elif trial_n == first_close + 1:
                ax = axarr[2]
                col = 'b'
            else:
                ax = axarr[3]
                col = self.mb_post_col
                postc += 1
                # if postc > 81: continue



            start, end = trial['overview_frame'], int(trial['overview_frame'] + trial['duration']*self.fps)
            try:
                end = np.where((trial["tracking_data"][start:, 0] > 440) & 
                    (trial["tracking_data"][start:, 0] < 560) & 
                    (trial["tracking_data"][start:, 1] > 775) & 
                    (trial["tracking_data"][start:, 1] < 910) )[0][0] + start
            except: continue
            if end - start > 20*self.fps: end = start + 20*self.fps

            arm, mean_speed = self.look_at_trial_in_detail(trial['tracking_data'][start:end])
            if arm is None: continue
            col = cc[arm]


            self.plot_tracking(trial['tracking_data'][start:end], mode="plot", color=col, linewidth=5,
                                alpha=.75, ax=ax, background=None)
            
        print(prec, postc)


    def plot_escapes_mbv2_good(self):
        f, ax = plt.subplots()

        self.data = (self.data & "uid='{}'".format(self.good_mb_uid))

        data = pd.DataFrame(self.data.fetch())
        data = data.sort_values("overview_frame")

        #440, 560 - 775, 910
        for i, (n, trial) in enumerate(data.iterrows()):
            if i in self.good_mb_exclude_trials or i > 8: continue

            if i < self.good_mb_closed:
                col = self.mb_pre_col
                alpha = .5
            elif i == self.good_mb_closed:
                col = "r"
                alpha = 1
            else:
                col = self.mb_post_col
                alpha = .5

            start, end = trial['overview_frame'], int(trial['overview_frame'] + trial['duration']*self.fps)

            end = np.where((trial["tracking_data"][start:, 0] > 440) & 
                (trial["tracking_data"][start:, 0] < 560) & 
                (trial["tracking_data"][start:, 1] > 775) & 
                (trial["tracking_data"][start:, 1] < 910) )[0][0] + start


            if end - start > 20*self.fps: end = start + 20*self.fps

            self.plot_tracking(trial['tracking_data'][start:end], mode="plot", color=col, linewidth=5,
                                alpha=alpha, ax=ax, background=self.mb_template)

            print(np.max(trial['tracking_data'][start:end, 1]))
        plt.show()

    def plot_escapes(self):
        af, aax = plt.subplots()
        sf, sax = plt.subplots()

        aax.set(xlim=[0, 1000], ylim=[0, 1000])
        sax.set(xlim=[0, 1000], ylim=[0, 1000])
        acount , scount = 0,  0
        for i, trial in enumerate(self.data):
            if i > 350: break
            print(trial['recording_uid'])
            if trial['duration'] == -1.0: continue
            if trial['experiment_name'] not in self.asym_experiments and trial['experiment_name'] not in self.sym_experiments: continue

            if trial['experiment_name'] in self.asym_experiments:
                ax = aax
                col = self.asym_color
                template = self.asym_template
                acount += 1
                if acount > 78: continue  # ? to make sure its the same number as the sym maze exp
            else:
                ax = sax
                col = self.sym_color
                template = self.sym_template
                scount += 1

            # Find when the mouse gets to the shelter
            start, end = trial['overview_frame'], int(trial['overview_frame'] + trial['duration']*self.fps)
            try:
                end = np.min(np.where(trial['tracking_data'][start:, -1]==0)) + start
            except: continue
            if (end-start)/self.fps < 2 or (end-start)/self.fps > 10: continue

            # check if it was on catwalk
            if trial['tracking_data'][start, 1] < 240: continue

            # check if it went on central pltf
            # if 15 in trial['tracking_data'][start:end, -1] or 16 in trial['tracking_data'][start:end, -1]: continue

            self.plot_tracking(trial['tracking_data'][start:end], mode="plot", color=col, linewidth=5,
                                alpha=.8, ax=ax, background=None)

        print(acount, scount)
        plt.show()




if __name__ == "__main__":
    p = Plotter()
    p.plot_escapes_mbv2()
    plt.show()