import sys
sys.path.append("./")


from Utilities.imports import *

from Processing.trials_analysis.trial_data_loader import TrialsLoader
from database.database_fetch import get_maze_template


# TODO stimuli that happen within N seconds should be counted as one trial -> important

class Analyser(TrialsLoader):
    def __init__(self):
        TrialsLoader.__init__(self)

        self.experiment = "Model Based V2"
        self.get_trials_by_exp(self.experiment)

        self.save_folder = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\MBV2"


        self.iterate_recordings()


    def setup_figure(self, n_trials):
        self.f, axarr = plt.subplots(ncols=int(np.ceil((1+n_trials)/3)), nrows=3, figsize=(25, 14))
        self.axarr = axarr.flatten()




    def iterate_recordings(self):
        for rec in self.recordings:
            if rec['recording_uid'] not in self.recording_uids: continue # We probs didn-t have stimuli for this            

            # Get tracking and  data for this recording
            tracking = self.get_recording_tracking(rec['recording_uid'], camera="overview", bp="snout").fetch1("tracking_data")
            data = self.get_rec_data(rec['recording_uid'])
           
            # Check if there were any stimuli in this session otherwise plot stuff
            if data.fetch("duration")[0] == -1: continue

            # Time filter tracking data
            tracking = self.filter_tracking(tracking)

            # Set up plotting
            self.setup_figure(len(data))
            template = np.rot90(get_maze_template(exp = self.experiment), 2)

            # Get the exploration tracking data and plot
            exploration_tracking = self.get_tracking_between_frames(tracking, 0, data.fetch("overview_frame")[0])
            self.plot_tracking(exploration_tracking, background=template, ax=self.axarr[0], title="exploration", s=5, alpha=.2)

            # Loop over each stimulus
            end_prev_trial = 0
            for n, stim in enumerate(data):
                # ? Plot ITI tracking
                if n > 0:
                    # Get the tracking between the end of the previous trial and the start of this and plot it 
                    iti_tracking = self.get_tracking_between_frames(tracking, end_prev_trial, stim['overview_frame']-1)
                    self.plot_tracking(iti_tracking, background=template, ax=self.axarr[n+1], s=5, alpha=1, with_time=False, color=[.4, .4, .4])

                
                # ? PLot escape tracking
                trial_end = stim["overview_frame_off"]+ 20*30 # ! Arbitrary and manual
                escape_tracking = self.get_tracking_between_frames(tracking,  stim['overview_frame'], trial_end)
                if n == 0: bg = template
                else: bg = None
                self.plot_tracking(escape_tracking, ax=self.axarr[n+1], background=bg, s=10, alpha=1, with_time=False, color="r", title="Trial - {}".format(n+1))


                # Keep track of when trial eneded
                end_prev_trial = trial_end



            # save and close image
            self.f.savefig(os.path.join(self.save_folder, rec["recording_uid"]+".png"))
            plt.close(self.f)
            a = 1





if __name__ == "__main__":
    a = Analyser()





