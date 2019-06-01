import sys
sys.path.append("./")


from Utilities.imports import *

from Processing.trials_analysis.TrialsMainClass import TrialsLoader
from database.database_fetch import get_maze_template

mpl.rcParams['image.aspect'] = "equal"
np.warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings("ignore")

class Analyser(TrialsLoader):
    def __init__(self):
        TrialsLoader.__init__(self)

        # ? min time between stimuli to be considered different trials
        self.fps = 40 
        self.isi_threshold = 60*self.fps  # ! hardcoded arbitrary threshold

        # For how long after the stim onset do we care
        self.after_stim_period = 20*self.fps

        self.skip_cage_time = 60*self.fps

        self.experiment = "Model Based V2"
        self.get_trials_by_exp(self.experiment)

        self.save_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\MBV2"
        self.save_folder_complete = None

        self.editor = Editor()

        # self.recordings = (self.recordings & "uid=282")

        self.iterate_recordings()


    def set_up_dir(self, rec_uid):
        self.save_folder_complete = os.path.join(self.save_folder, rec_uid)
        if not os.path.isdir(self.save_folder_complete):
            os.mkdir(self.save_folder_complete)

    def setup_figure(self, ncols, nrows, background):
        f, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 14))
        axarr = axarr.flatten()
        for ax in axarr:
        #     ax.imshow(background, cmap="Greys", origin="lower")
            ax.set(xticks=[], yticks=[])

        return f, axarr

    def group_trials(self, stim_data):
        data = pd.DataFrame(stim_data.fetch()).sort_values("overview_frame")
        isi = np.insert(np.diff(data.overview_frame), 0, 10000) # ? inserting a very high value for the first one to make sure is counted

        if np.any(np.where(isi < self.isi_threshold)): # ! hardcoded arbitrary threshold
            to_keep = np.where(isi >= self.isi_threshold)[0]
            data = data.iloc[to_keep]
            data.index = np.arange(len(data))
        return data

    def plot_exploration(self, tracking, template, recuid):
        f, axarr = plt.subplots(ncols=2, nrows=2)
        axarr= axarr.flatten()
        # Plot tracking
        self.plot_tracking(tracking, mode="time", ax=axarr[0], title="Exploration", background=template)

        # Plot heatmap
        self.plot_tracking(tracking, ax=axarr[1], mode="hexbin", mincnt=1, bins="log")

        speed = calc_distance_between_points_in_a_vector_2d(tracking)
        dist_covered = int(round(np.sum(speed)))
        pc = percentile_range(speed)

        axarr[1].text(.4, .05, "Distance covered: {}".format(dist_covered), verticalalignment='bottom', horizontalalignment='right',
                        transform=axarr[1].transAxes, color='white', fontsize=10)
        axarr[1].text(.9, .05, "Speed: median:{}, low/high:{}/{}".format(round(pc.median,2), round(pc.low,2), round(pc.high,2)), verticalalignment='bottom', horizontalalignment='right',
                        transform=axarr[1].transAxes, color='white', fontsize=10)

        axarr[1].set(title="Occupancy", xticks=[], yticks=[])

        # Plot speed heatmap
        axarr[2].scatter(tracking[:, 0], tracking[:, 1], c=tracking[:, 2], alpha=.05, s=300, cmap="Reds")

        # Plot prob of being on each maze component
        rois = self.get_rec_maze_components(recuid)
        p_occupancy = []
        for i in rois.roi_index.values:
            p_occupancy.append(tracking[:, -1][(tracking[:, -1] == i)].sum() / len(tracking))

        rois['p_occupancy'] = p_occupancy
        axarr[3].imshow(template, cmap="Greys", origin="lower")
        for i, row in rois.iterrows():
            if row.name == "b15": color="b"
            else: color="r"
            try:
                axarr[3].scatter(row.position[0], row.position[1],c=color,  s=row.p_occupancy*400)
            except: pass
        
        self.save_figure(f, "exploration")

    def save_figure(self, f, savename):
        f.savefig(os.path.join(self.save_folder_complete, savename+".png"))
        plt.close(f)

    def get_time_on_L_average(self, tracking, n_seconds):
        on_l = tracking.copy()
        on_l[(on_l[:, 0]>400)&(on_l[:, 0]<600)&(on_l[:, 1]>640)&(on_l[:, 1]<780)] = np.nan
        mask = (np.isnan(on_l))
        on_l = tracking.copy()
        on_l[~mask] = 0
        on_l[mask] = 1
        on_l = on_l[:, 0]
        # return moving_average(on_l, n_seconds*self.fps)
        return line_smoother(on_l, window_size=5001)

    def get_on_l_tracking(self, tracking):
        x_inflated_tr = tracking.copy()
        x_inflated_tr[:, 0] = tracking[:, 0] + np.arange(len(tracking))

        tr = tracking.copy()
        tr[tr[:, 0]<400] = np.nan
        tr[tr[:, 0]>600] = np.nan
        tr[tr[:, 1]>780] = np.nan
        tr[tr[:, 1]<640] = np.nan
        tr[:, 0] = tr[:, 0] + np.arange(len(tracking))
        return x_inflated_tr, tr

    def plot_iti(self, n, axarr, tracking, stim, template, exploration_tracking, end_prev_trial):
        if n > 0:
            # Get the tracking between the end of the previous trial and the start of this and plot it 
            iti_tracking = self.get_tracking_between_frames(tracking[1], end_prev_trial, stim['overview_frame']-1)
            self.plot_tracking(iti_tracking, background=template, mode="time", s=50, alpha=1, ax=axarr[1])
            speed = calc_distance_between_points_in_a_vector_2d(iti_tracking)
        else:
            self.plot_tracking(exploration_tracking, background=template, mode="time", s=50, alpha=1, ax=axarr[1])
            iti_tracking = exploration_tracking.copy()

        # Y axis during exploration
        iti_tracking, iti_L_tracking = self.get_on_l_tracking(iti_tracking)
        self.plot_tracking(iti_tracking, mode="time", title="ITI + time", ax=axarr[2], alpha=.8)
        self.plot_tracking(iti_L_tracking, mode="scatter",c="r", ax=axarr[2], label="on Lambda")
        axarr[2].set(ylim=[0, 1000])


        # ? annotate ITI plot
        iti_dur, _ = divmod((stim['overview_frame'] - end_prev_trial)/self.fps, 60)
        axarr[1].set(title="ITI: {}min".format(round(iti_dur)))

        speed = calc_distance_between_points_in_a_vector_2d(exploration_tracking)

        dist_covered = int(round(np.sum(speed)))
        axarr[1].text(.4, .05, "Distance covered: {}".format(dist_covered), verticalalignment='bottom', horizontalalignment='right',
                        transform=axarr[1].transAxes, color='white', fontsize=10)


    def iterate_recordings(self):
        bps = ['snout', 'body', 'tail_base']
        for rec in self.recordings:
            if rec['recording_uid'] not in self.recording_uids: continue # We probs didn-t have stimuli for this            
            # Get tracking and  data for this recording
            tracking = [self.get_recording_tracking(rec['recording_uid'], camera="overview", bp=bp).fetch1("tracking_data") for bp in bps]
            data = self.get_rec_data(rec['recording_uid'])
           
            # Check if there were any stimuli in this session otherwise plot stuff
            if data.fetch("duration")[0] == -1: 
                continue
            else: print("Processing: ", rec['recording_uid'])

            # Group stimuli that happend close in time into the same trial
            data = self.group_trials(data)

            # Set up plotting
            self.set_up_dir(rec['recording_uid'])
            template = np.rot90(get_maze_template(exp = self.experiment), 2)
            cap = self.get_opened_video(rec['recording_uid'])

            # ? get time binned p(L)
            # on_l_avg = self.get_time_on_L_average(tracking[1], 300)
            # highlight_on_l = self.get_on_l_tracking(tracking[1])
            x_inflated_Tracking, x_inflated_l_highlated_tracking = self.get_on_l_tracking(tracking[1])
            x_inflated_l_highlated_tracking_up, x_inflated_l_highlated_tracking_down = x_inflated_l_highlated_tracking.copy(), x_inflated_l_highlated_tracking.copy()
            x_inflated_l_highlated_tracking_up[np.insert(np.diff(x_inflated_l_highlated_tracking[:, 1])<0, 0, 0), :] = np.nan
            x_inflated_l_highlated_tracking_down[np.insert(np.diff(x_inflated_l_highlated_tracking[:, 1])>0, 0, 0), :] = np.nan

            # Get the exploration tracking data and plot
            exploration_tracking = self.get_tracking_between_frames(tracking[1], self.skip_cage_time, data.overview_frame[0])
            self.plot_exploration(exploration_tracking, template, rec['recording_uid'])

            # Loop over each stimulus
            end_prev_trial = 0
            prev_escapes = []
            for n, stim in data.iterrows():
                print("     ... trial: ", n+1)
                f, axarr = self.setup_figure(ncols=3, nrows=3, background=template)
                
                # ? Plot ITI tracking
                self.plot_iti(n, axarr, tracking, stim, template, exploration_tracking, end_prev_trial)


                # ? Get frame at start of trial
                frame = self.get_average_frames(cap, stim['overview_frame'], 30)
                if frame is not None:
                            axarr[0].imshow(frame)

                # ? plot prev escapes
                if prev_escapes:
                    for esc in prev_escapes:
                        self.plot_tracking(esc, mode="plot", alpha=.8, color=None, ax=axarr[8], linewidth=8, background=template)
                    axarr[8].set(title="Previous escapes")

                # ? preparare escape tracking
                trial_end = stim["overview_frame_off"]+ self.after_stim_period
                escape_tracking = [self.get_tracking_between_frames(t,  stim['overview_frame'], trial_end)  for t in tracking]
                prev_escapes.append(escape_tracking[1])

                # ? Plot escape
                self.plot_tracking(escape_tracking[1], mode="plot", alpha=.75, background=template, 
                                color="r", ax=axarr[3], linewidth=8,
                                title="Trial - {}".format(n+1))

                for ax in axarr[6:8]: # ? Plot 3 body parts
                    self.plot_tracking_2bp(escape_tracking[0],escape_tracking[1],  ax=ax, background=template, )


                # ? Plot p(L)
                self.plot_tracking(x_inflated_Tracking, mode="time", title="Time on L for whole session", ax=axarr[5], alpha=.5, s=10)
                self.plot_tracking(x_inflated_l_highlated_tracking_up, mode="scatter",c="r", ax=axarr[5], label="Towards THREAT")
                self.plot_tracking(x_inflated_l_highlated_tracking_down, mode="scatter",c="blue", ax=axarr[5], label="Towards SHELTER")

                axarr[5].set(ylim=[0, 1000])

                # Mark when stims happend
                for s in data.overview_frame:
                    if s == stim.overview_frame: color="green"
                    else: color="white"
                    axarr[5].axvline(s, color=color, linewidth=2, alpha=.5)

                # ? stuff
                # Keep track of when trial eneded
                end_prev_trial = trial_end
     
                # Edit axes
                axarr[6].set(title="Threat Platform", xlim=[420, 600], ylim=[90, 425])
                axarr[7].set(title="Intermediate Platform", xlim=[420, 600], ylim=[500, 670])

                for ax in axarr:
                    ax.legend()

                # plt.show()

                # save fig
                self.save_figure(f, "Trial-{}".format(n+1))









if __name__ == "__main__":
    a = Analyser()





