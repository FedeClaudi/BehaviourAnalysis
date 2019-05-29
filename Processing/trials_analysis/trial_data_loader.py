import sys
sys.path.append("./")


from Utilities.imports import *

from scipy import signal

class TrialsLoader:
    def __init__(self):
        # Load all the stimul with the recording and session information
        self.data = (Recording * Session * Stimuli)
        self.recordings = (Recording * Session)

        self.get_experiments()
        self.get_recording_uids()

    def get_trials_by_exp(self, exp):
        if isinstance(exp, list): raise NotImplementedError("doesnt work with list of exp") # TODO make it work with lits
        elif exp not in self.experiments: raise ValueError("experiment name not in table")
        else:
            self.data =  (self.data & "experiment_name='{}'".format(exp))
            self.recordings =  (self.recordings & "experiment_name='{}'".format(exp))

    def get_experiments(self):
        self.experiments = sorted(set(self.data.fetch("experiment_name")))

    def get_recording_uids(self):
        self.recording_uids = sorted(set(self.data.fetch("recording_uid")))

    def get_recording_tracking(self, rec_uid, camera = None, bp=None):
        if camera is None:
            if bp is None:
                return (TrackingData.BodyPartData & "recording_uid='{}'".format(rec_uid))
            else:
                return (TrackingData.BodyPartData & "recording_uid='{}'".format(rec_uid) & "bpname='{}'".format(bp))
        else:
            if bp is None:
                return (TrackingData.BodyPartData & "recording_uid='{}'".format(rec_uid) & "camera='{}'".format(camera))
            else:
                return (TrackingData.BodyPartData & "recording_uid='{}'".format(rec_uid) & "camera='{}'".format(camera)  & "bpname='{}'".format(bp))

    def get_rec_data(self, rec_uid):
        return (self.data & "recording_uid='{}'".format(rec_uid))

    @staticmethod
    def get_tracking_between_frames(tracking, f0, f1):
        if f1 == -1: return None 
        return tracking[f0:f1, :]

    @staticmethod
    def plot_tracking(tracking, ax=None, background=None, with_time=True, title=None, color=None, scatter=False, s=50, alpha=.8):
        if ax is None: 
            f,ax = plt.subplots()

        if background is not None:
            ax.imshow(background, cmap="Greys", origin="lower")
        
        if with_time:
            ax.scatter(tracking[:, 0], tracking[:, 1], c=np.arange(tracking.shape[0]), alpha=alpha, s=s)
        else:
            if scatter:
                ax.scatter(tracking[:, 0], tracking[:, 1], c=color, alpha=alpha, s=s)
            else:
                ax.plot(tracking[:, 0], tracking[:, 1], color=color, alpha=alpha)

        if title:
            ax.set(title=title)

        ax.set(xticks=[], yticks=[])

    def filter_tracking(self, tracking, window_len=31):
        filtered = np.vstack([signal.medfilt(tracking[:, i], window_len) for i in range(tracking.shape[1])]).T

        filtered[tracking[:, 2] > 1] = np.nan

        # f, ax = plt.subplots()
        # self.plot_tracking(tracking, color='k', ax=ax, with_time=False, scatter=True)
        # self.plot_tracking(filtered, color='r', ax=ax, with_time=False, scatter=True)
        # plt.show()

        return filtered
        


if __name__ == "__main__":
    TrialsLoader()