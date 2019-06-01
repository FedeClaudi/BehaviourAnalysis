import sys
sys.path.append("./")


from Utilities.imports import *

from scipy import signal

class TrialsLoader:
    def __init__(self):
        # Load all the stimul with the recording and session information
        self.data = (Recording * Session * Stimuli * TrackingData.BodyPartData)
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
    def plot_tracking(tracking, mode=None, ax=None, background=None,  title=None,  **kwargs):
        if ax is None: 
            f,ax = plt.subplots()

        if background is not None:
            ax.imshow(background, cmap="Greys", origin="lower")
        
        if mode == "time":
            ax.scatter(tracking[:, 0], tracking[:, 1], c=np.arange(tracking.shape[0]), **kwargs)
        elif mode == "scatter":
            ax.scatter(tracking[:, 0], tracking[:, 1],**kwargs)
        elif mode == "plot":
            ax.plot(tracking[:, 0], tracking[:, 1], **kwargs)
        elif mode == "hexbin":
            ax.hexbin(tracking[:, 0], tracking[:, 1], **kwargs)

        if title:
            ax.set(title=title)
        ax.set(xticks=[], yticks=[])

    @staticmethod
    def plot_tracking_3bp(t1, t2, t3, interval = 10, background=None, ax=None, c1=None, c2=None, title="", **kwargs):
        if c1 is None: c1 =  'red'
        if c2 is None: c2 = "green"


        if ax is None:
            f, ax = plt.subplots()
        if background is not None:
            ax.imshow(background, cmap="Greys", origin="lower")
        if title: ax.set(title=title)

        markers = ["p", "D"]
        for t, symb, c, s in zip([t1], markers, [c1,c2], [400, 150]):
            ax.scatter(t[0:-1:interval, 0], t[0:-1:interval, 1], c=c, alpha=1,  s=s, marker=symb)

        ax.plot([t1[0:-1:interval, 0], t2[0:-1:interval, 0]], [t1[0:-1:interval, 1], t2[0:-1:interval, 1]], color=c1, alpha=.8, linewidth=5)
        ax.plot([t2[0:-1:interval, 0], t3[0:-1:interval, 0]], [t2[0:-1:interval, 1], t3[0:-1:interval, 1]], color=c2, alpha=.8, linewidth=4)

    @staticmethod
    def plot_tracking_2bp(t1, t2, interval=10, background=None, ax=None, title=None):
        if ax is None:
            f, ax = plt.subplots()
        if background is not None:
            ax.imshow(background, cmap="Greys", origin="lower")
        if title: ax.set(title=title)

        ax.plot([t1[0:-1:interval, 0], t2[0:-1:interval, 0]], [t1[0:-1:interval, 1], t2[0:-1:interval, 1]], color="blue", alpha=.8, linewidth=3)
        ax.plot(t1[:, 0], t1[:, 1], color="red", linewidth=6, alpha=.6)
        ax.plot(t2[:, 0], t2[:, 1], color="green", linewidth=4, alpha=.6)

    def filter_tracking(self, tracking, window_len=31):
        filtered = np.vstack([signal.medfilt(tracking[:, i], window_len) for i in range(tracking.shape[1])]).T

        filtered[tracking[:, 2] > 1] = np.nan

        # f, ax = plt.subplots()
        # self.plot_tracking(tracking, color='k', ax=ax, with_time=False, scatter=True)
        # self.plot_tracking(filtered, color='r', ax=ax, with_time=False, scatter=True)
        # plt.show()

        return filtered

    def get_opened_video(self, recuid):
        filepaths = (Recording.FilePaths & "recording_uid='{}'".format(recuid)).fetch1()
        cap = cv2.VideoCapture(filepaths['overview_video'])
        return cap

    def get_rec_maze_components(self, recuid):
        sess_name = (Recording & "recording_uid='{}'".format(recuid)).fetch1("session_name")
        rois = pd.DataFrame((MazeComponents & "session_name='{}'".format(sess_name)).fetch())
        del rois['uid'], rois['session_name']
        rois_ids = {p:i for i,p in enumerate(rois.keys())}

        rois = rois.T
        rois.columns = ["position"]
        rois['roi_index'] = rois_ids.values()
        return rois

    def get_average_frames(self, cap, start_frame, n_frames):
        try:
            frame = self.editor.get_selected_frame(cap, start_frame)
        except:
            self.editor = Editor()
            frame = self.editor.get_selected_frame(cap, start_frame)

        if frame is None: return frame

        avg = np.float32(frame)
        for i in range(n_frames):
            nframe = self.editor.get_selected_frame(cap, start_frame+(i*5)) # ! skipping frames
            try:
                cv2.accumulateWeighted(nframe, avg, 0.75) 
            except:
                return None
            res = cv2.convertScaleAbs(avg)

        return res

    def save_data_as_df(self, savepath):
        pd.DataFrame(self.data.fetch()).to_pickle(savepath)
        

if __name__ == "__main__":
    TrialsLoader()