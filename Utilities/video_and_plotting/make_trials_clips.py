import sys
sys.path.append('./') 

import os 
import cv2
from tqdm import tqdm
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from collections import namedtuple
from functools import partial

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor

from Processing.plot.plotting_utils import *
from Processing.plot.video_plotting_toolbox import *
from skimage.transform import resize


from database.database_fetch import *

from database.TablesDefinitionsV4 import *

class TrialClipsMaker(Editor):
    """[This class creates one clip with all the stimuli (trial) in that recording. 
       offers the possibility to overlay tracking data, add threat video frames...]
    
    Arguments:
        Editor {[type]} -- [description]
    
    Raises:
        NotImplementedError: [description]
    """
    def __init__(self, process_recs_in_range=None, add_threat_video=False, overlay_pose=False, overlay_text=False, clean=False, save_fld=None):
        """[Creates a clip with all the trial for one recording]
        
        Keyword Arguments:
            process_recs_in_range {[list]} -- [list of integers, only process recordings with UID in range of list] (default: {None})
            add_threat_video {bool} -- [add threat video to clip, default only shows overview video] (default: {False})
            overlay_pose {bool} -- [overvlay DLC tracking on videos] (default: {False})
            overlay_text {bool} -- [overlay text over the frame (e..g elapsed time)] (default: {False})

        """
        Editor.__init__(self)

        # Get all the saved videos in the dstination folder
        if save_fld is None:
            self.save_folder = self.paths['trials_clips']
        else:
            if not os.path.isdir(save_fld): raise FileNotFoundError
            self.save_folder = save_fld
        self.clips_in_save_folder = [f for f in os.listdir(self.save_folder)]

        # Get the relevant tables from the database
        self.prep_data()

        # Get a few other relevant variables
        self.process_recs_in_range = process_recs_in_range
        self.add_threat_video = add_threat_video
        self.overlay_pose = overlay_pose
        self.overlay_text = overlay_text
        self.clean = clean


        # Define parameters for decorations on video
        if not clean:
            self.video_decoration_params = {
                "pre_stim_interval": 1,
                "post_stim_interval": 20, # ? number of seconds before and after the stimulus to include in the clip
                "border_size":20,
                "color_on": [100, 255, 100],
                "color_off": [20,20,20],
            }
        else:
            self.video_decoration_params = {
                "pre_stim_interval": 5,
                "post_stim_interval": 20, # ? number of seconds before and after the stimulus to include in the clip
                "border_size":0,
                "color_on": [100, 255, 100],
                "color_off": [20,20,20],
            } 
            self.overlay_pose = False
            self.overlay_text = False


        # get a copy of the original params to restore them if they get changed during processing
        self._add_threat_video = add_threat_video
        self._overlay_pose = overlay_pose


    def restore_settings(self):
        self.add_threat_video = self._add_threat_video
        self.overlay_pose = self._overlay_pose


    def prep_data(self):
        """[Combine DJ tables to have all the necessary info in the same place]
        """
        self.data = (Recording * Stimuli)


    def loop_over_recordings(self, skip_every=None):
        """[ Loop over all recordings in table, see which one needs processing and extract the corresponding stimuli,
        file paths and metadata]
        
        Raises:
            NotImplementedError: [Doesnt work for behaviour software]
            NotImplementedError: [doesnt work for visual stims]
        """
        r = namedtuple("r", "recording_uid uid session_name software ai_file_path")

        for recn, recuid in enumerate(sorted(set(self.data.fetch("recording_uid")))):
            if skip_every is not None:
                if not recn % skip_every == 0: continue

            self.restore_settings()
            
            _rec = (Recording & "recording_uid='{}'".format(recuid)).fetch1()
            rec = r(*_rec.values())

            del _rec['software']
            del _rec["ai_file_path"]

            # Check if needs to be processed
            if self.process_recs_in_range is not None:
                if rec.uid < self.process_recs_in_range[0] or rec.uid > self.process_recs_in_range[1]: continue
            print("Processing recording: ", rec.recording_uid)

            # Check if a video for this record exists already
            self.video_savepath = os.path.join(self.save_folder, rec.recording_uid + "_all_trials.mp4")
            if os.path.isfile(self.video_savepath): #? check that a video doesn't already exists. if it does but its empty overwrite it
                if os.path.getsize(self.video_savepath) > 10000: continue

            # Get the stimuli for this recording
            if rec.software == 'behaviour':
                continue
            else:
                self.stimuli = pd.DataFrame((self.data & _rec).fetch())

            if len(self.stimuli) == 0:
                warnings.warn("no stimuli found for {}".format(rec.recording_uid))
                continue

            if "video" in self.stimuli.stim_type or "visual" in self.stimuli.stim_type: 
                warnings.warn("Not implement for visual stimuli ??")
                continue

            # Get videopath for this recording
            self.rec_paths = pd.DataFrame(Recording.FilePaths & _rec)

            # Get the aligned overview and threat frames if they exist
            self.aligned_frame_times = pd.DataFrame(Recording.AlignedFrames & _rec)

            # # if there is no info about the aligned frame times we cannot add the threat video
            if not len(self.aligned_frame_times) and self.add_threat_video:
                warnings.warn("\n   cannot add threat video if the frames haven't been aligned, populate FrameTimes")
                self.add_threat_video = False

            # get tracking data 
            if self.overlay_pose:
                self.overview_tracking = (TrackingData & _rec).fetch()
                if not len(self.overview_tracking): 
                    warnings.warn("\n Could not find tracking data for this recording")
                    self.overlay_pose = False

                elif self.add_threat_video:
                    self.threat_tracking = get_tracking_given_recuid(rec.recording_uid, cam="threat")
                    if not len(self.threat_tracking): self.threat_tracking = None
                    # TODO need to create and fill a tracking data table for the threa vids first

            # Set up to write this recording's clip
            ret = self.setup_clip_writing()
            if ret:
                self.prep_circles()
                self.create_clip()

    def setup_clip_writing(self):
        """[Create an opencv write with the correct parameters]
        """
        # define named tuple to hold video params
        vparams = namedtuple("vparams", "nframes width height fps")

        # overview video params
        overview_video_path = self.rec_paths.overview_video.values[0]
        self.overview_cap = cv2.VideoCapture(overview_video_path)
        self.overview_params = vparams(*self.get_video_params(self.overview_cap))

        if self.overview_params.nframes == 0: return False

        # threat video params
        if self.add_threat_video:
            try:
                threat_video_path = self.rec_paths.threat_video.values[0]
            except:
                self.add_threat_video = False
                self.frame_shape = [int(self.overview_params.height), int(self.overview_params.width)]
            else:
                self.threat_cap = cv2.VideoCapture(threat_video_path)
                self.threat_params = vparams(*self.get_video_params(self.threat_cap))

                if self.threat_params.fps < 1:    # ? smth went wrong when opening the threat cap - likely video has no frames
                    self.add_threat_video = False
                    self.frame_shape = [int(self.overview_params.height), int(self.overview_params.width)]
                else:
                    # if we are placing the two vids side by side, we need to scale the overview video to be as tall as the threat video
                    self.height_ratio = self.threat_params.height / self.overview_params.height

                    if self.height_ratio < 1: raise ValueError("Error calculating scaling factor.")

                    self.frame_shape = [int(self.threat_params.height), int(self.overview_params.width * self.height_ratio + self.threat_params.width)]
        else:
            self.frame_shape = [int(self.overview_params.height), int(self.overview_params.width)]

        # open the opencv writer
        self.writer = self.open_cvwriter(self.video_savepath, 
                                        w=self.frame_shape[1]+self.video_decoration_params['border_size']*2,
                                        h=self.frame_shape[0]+self.video_decoration_params['border_size']*2,
                                        framerate = int(self.overview_params.fps), iscolor=True)

        return True

    @staticmethod
    def get_selected_frame(cap, show_frame):
        """[return a specific frame from a video ]
        
        Arguments:
            cap {[opencv Cap object]} -- [video object]
            show_frame {[int]} -- [frame number to be shown]
        
        Returns:
            [np.array] -- [frame]
        """
        cap.set(1, show_frame)
        ret, frame = cap.read() # read the first frame
        if ret: return frame
        else: return ret 

    @staticmethod
    def get_threat_frame(frame_times, framen):
        oframes = frame_times.overview_frames_timestamps.iloc[0]
        tframes = frame_times.threat_frames_timestamps.iloc[0]
        ifi = np.mean(np.diff(oframes))

        close_tframes = [i for i, x in enumerate(tframes) if abs(x-oframes[framen])<=ifi]
        if close_tframes:
            return close_tframes[0]
        else:
            return False


    def prep_circles(self):
        """[Show some circles on the frame to show the total number of stimuli and the current displayed one,
            need to prepare the parameters for number of circles, size, position... ]
        """
        n_squares = len(self.stimuli)
        centers = np.linspace(20, self.frame_shape[1]-20, n_squares)

        complete_centers = [(int(c), int(self.frame_shape[0] - (self.frame_shape[0]*.95))) for c in centers]

        if len(centers)>1:
            radius = int(np.diff(centers)[0]/3)
        else: radius = 20
        if radius > 20: radius = 20
        self.stimuli_display_circles = [complete_centers, radius]

    def create_clip(self):
        if self.add_threat_video:
            frame_timestamps = self.aligned_frame_times.aligned_frame_timestamps.values[0]

        print("     creating videoclip")
        for stim_number, stim in self.stimuli.iterrows():
            if stim.overview_frame == -1: continue   # ? it was a place holder entry in the table, there were no stims for that session
            # Loop over each stimulus and add the corresponding section of the video to the main video
            print('           ... adding new trial to the clip')

            # Get start and stop frames for the overview video
            # ? The corresponding threat videos need to be pulled by the aligned frame time table
            clip_start = int(stim.overview_frame - self.video_decoration_params['pre_stim_interval']*self.overview_params.fps)
            clip_end = int(stim.overview_frame + (stim.duration*self.overview_params.fps) + self.video_decoration_params['post_stim_interval']*self.overview_params.fps)
            clip_number_of_frames = int(clip_end - clip_start)

            # move overview cap to the frame at the start of this clip
            _ = self.get_selected_frame(self.overview_cap, clip_start-1)

            # Keep reading frames until within post stim
            for frame_counter in range(clip_number_of_frames):
                ret, frame = self.overview_cap.read()
                if not ret: break
                frame_number = clip_start + frame_counter  # 

                # Get the threat video frame
                if self.add_threat_video:
                    threat_frame_number = self.get_threat_frame(self.aligned_frame_times, frame_number)
                    if threat_frame_number:
                        threat_frame = self.get_selected_frame(self.threat_cap, threat_frame_number)
                    else:
                        threat_frame = np.zeros((self.threat_params.height, self.threat_params.width, 3), dtype=frame.dtype)  # ? black frame if none is found

                # TODO overlay tracking data
                # TODO don't just show the tracking data, also show a trace of the trajectory
                # TODO show also what the mouse did in the ITI? MAYBE BABE
                if self.overlay_pose:
                    warning.warn("not implemented overlay tracking on frame")

                # ? Combine overview and threat frames
                    # rescale overview frame
                    frame = cv2.resize(frame, None, fx = self.height_ratio, fy = self.height_ratio, interpolation = cv2.INTER_CUBIC)

                    # put overview and therat in the same frame
                    try:
                        frame = np.hstack([frame, threat_frame])
                    except:
                        a = 1

                # Check if the stimulus is on or off
                if frame_number < clip_start+self.video_decoration_params['pre_stim_interval'] or frame_number >clip_end-self.video_decoration_params['post_stim_interval']:
                    sign = ''
                    self.curr_color = self.video_decoration_params['color_off']  # ? stim off
                else:
                    sign = '+'
                    self.curr_color = self.video_decoration_params['color_on'] # ? stim on

                # add colored border to the frame
                frame = cv2.copyMakeBorder(frame, self.video_decoration_params['border_size'], 
                                                self.video_decoration_params['border_size'], 
                                                self.video_decoration_params['border_size'], 
                                                self.video_decoration_params['border_size'],
                                                 cv2.BORDER_CONSTANT, value=self.curr_color)
                        
                if self.overlay_text:
                    # add elapsed time and so on...
                    frame_time = (frame_number - clip_start) / self.overview_params.fps
                    frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                    cv2.putText(frame, sign + str(frame_time) + 's', (self.frame_shape[1] - 120, self.frame_shape[0] - 100), 0, 1,
                                (20, 255, 20), thickness=2)


                    for i, center in enumerate(self.stimuli_display_circles[0]):
                        if i == stim_number:
                            color = [200, 50, 50]
                            border = -1
                        else:
                            color = [255, 200, 200]
                            border = 5
                        cv2.circle(frame, (center[0], center[1]), self.stimuli_display_circles[1], color, border)

                # Save to file
                self.writer.write(frame)
            break



if __name__ == "__main__":
    tcm = TrialClipsMaker(process_recs_in_range = [40, 1000], 
                            add_threat_video    = False, 
                            overlay_pose        = False, 
                            overlay_text        = False,
                            clean=True, 
                            save_fld="Z:\\branco\\Federico\\raw_behaviour\\maze\\_overview_training_clips")
    tcm.loop_over_recordings(skip_every=10)

