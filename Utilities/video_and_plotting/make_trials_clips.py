import sys
sys.path.append('./') 

import os 
import cv2
from tqdm import tqdm
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from functools import partial

from Utilities.file_io.files_load_save import *
from Utilities.video_and_plotting.video_editing import Editor
from database.NewTablesDefinitions import *
from Processing.plot.plotting_utils import *
from Processing.plot.video_plotting_toolbox import *
from skimage.transform import resize


from database.database_fetch import *


class TrialClipsMaker(Editor):
    """[This class creates one clip with all the stimuli (trial) in that recording. 
       offers the possibility to overlay tracking data, add threat video frames...]
    
    Arguments:
        Editor {[type]} -- [description]
    
    Raises:
        NotImplementedError: [description]
    """
    def __init__(self, process_recs_in_range=None, add_threat_video=False, overlay_pose=False, overlay_text=False):
        """[Creates a clip with all the trial for one recording]
        
        Keyword Arguments:
            process_recs_in_range {[list]} -- [list of integers, only process recordings with UID in range of list] (default: {None})
            add_threat_video {bool} -- [add threat video to clip, default only shows overview video] (default: {False})
            overlay_pose {bool} -- [overvlay DLC tracking on videos] (default: {False})
            overlay_text {bool} -- [overlay text over the frame (e..g elapsed time)] (default: {False})

        """
        Editor.__init__(self)

        # Get all the saved videos in the dstination folder
        self.save_folder = self.paths['trials_clips']
        self.clips_in_save_folder = [f for f in os.listdir(self.save_folder)]

        # Get the relevant tables from the database
        self.recordings = pd.DataFrame(Recordings().fetch())
        self.behav_stims = BehaviourStimuli()
        self.mantis_stims = MantisStimuli()
        self.videofiles = pd.DataFrame(VideoFiles().fetch())
        self.all_aligned_frame_times = pd.DataFrame(FrameTimes().fetch())

        # Get a few other relevant variables
        self.process_recs_in_range = process_recs_in_range
        self.add_threat_video = add_threat_video
        self.overlay_pose = overlay_pose
        self.overlay_text = overlay_text

        # get a copy of the original params to restore them if they get changed during processing
        self._add_threat_video = add_threat_video
        self._overlay_pose = overlay_pose

        # Define parameters for decorations on video
        self.video_decoration_params = {
            "pre_stim_interval": 1,
            "post_stim_interval": 20, # ? number of seconds before and after the stimulus to include in the clip
            "border_size":20,
            "color_on": [100, 255, 100],
            "color_off": [20,20,20],
        }

    def restore_settings(self):
        self.add_threat_video = self._add_threat_video
        self.overlay_pose = self._overlay_pose

    def loop_over_recordings(self):
        """[ Loop over all recordings in table, see which one needs processing and extract the corresponding stimuli,
        file paths and metadata]
        
        Raises:
            NotImplementedError: [Doesnt work for behaviour software]
            NotImplementedError: [doesnt work for visual stims]
        """
        for i, rec in self.recordings.iterrows():
            self.restore_settings()

            # Check if needs to be processed
            if self.process_recs_in_range is not None:
                if rec.uid < self.process_recs_in_range[0] or rec.uid > self.process_recs_in_range[1]: continue
            print("Processing recording: ", rec.recording_uid)

            # Check if a video for this record exists already
            self.video_savepath = os.path.join(self.save_folder, rec.recording_uid + "_all_trials.mp4")
            if os.path.isfile(self.video_savepath): #? check that a video doesn't already exists. if it does but its empty overwrite it
                if os.path.getsize(self.video_savepath) > 10000: continue

            # Get the stimuli for this recording
            if rec['software'] == 'behaviour':
                continue
                # raise NotImplementedError("Doesnt work for behaviour vids")
                self.stimuli = pd.DataFrame([s for s in behav_stims if s['recording_uid']==rec['recording_uid']])
            else:
                self.stimuli = pd.DataFrame([s for s in self.mantis_stims if s['recording_uid'] == rec['recording_uid']]).sort_values("overview_frame")

            if "video" in self.stimuli.stim_type or "visual" in self.stimuli.stim_type: raise NotImplementedError

            # Get videopath for this recording
            self.rec_paths = self.videofiles.loc[self.videofiles.recording_uid == rec.recording_uid]

            # Get the aligned overview and threat frames if they exist
            self.aligned_frame_times = self.all_aligned_frame_times.loc[self.all_aligned_frame_times.recording_uid == rec.recording_uid]

            # # if there is no info about the aligned frame times we cannot add the threat video
            if not len(self.aligned_frame_times) and self.add_threat_video:
                warnings.warn("\n   cannot add threat video if the frames haven't been aligned, populate FrameTimes")
                self.add_threat_video = False

            # get video metadata
            self.video_metadata = get_videometadata_given_recuid(rec.recording_uid, just_fps=False)

            # get tracking data
            if self.overlay_pose:
                self.overview_tracking = get_tracking_given_recuid(rec.recording_uid)
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
        overview_video_path = self.rec_paths.loc[self.rec_paths.camera_name == "overview"].converted_filepath.iloc[0]
        self.overview_cap = cv2.VideoCapture(overview_video_path)
        self.overview_params = vparams(*self.get_video_params(self.overview_cap))

        if self.overview_params.nframes == 0: return False

        # threat video params
        if self.add_threat_video:
            try:
                threat_video_path = self.rec_paths.loc[self.rec_paths.camera_name == "threat"].converted_filepath.iloc[0]
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
            # break


"""    
    ==========================================================================================================================
    ==========================================================================================================================
    ==========================================================================================================================
    OBSOLETE BITS OF CODE DOWN HERE !!
    It works, but its not as flexible and powerfull as the class above, but keep it to make sure we have a working
    clip maker if smth goes wrong
    ==========================================================================================================================
    ==========================================================================================================================
    ==========================================================================================================================
    
"""

class ClipWriter:
    def __init__(self, videopath, stimuli, clean_vids):
        # stimuli is a dict containing stimuli start and finish frames
        self.stimuli = stimuli
        self.videopath = videopath
        _, name = os.path.split(self.videopath)
        savefld = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\trials_clips'
        self.savepath = os.path.join(savefld, name[:-4]+'_trials.mp4')

        if os.path.exists(self.savepath): return # Avoid overwriting

        self.clean_vids = clean_vids

        # Define params and open writer
        self.define_decoration_params()
        self.open_cv_writer()

        # Write videos
        try: 
            self.make_concatenated_clips()
        except:
            return

    def define_decoration_params(self):
        # parameters to draw on frame
        self.border_size = 20
        self.color_on = [100, 255, 100]
        self.color_off = [20,20,20]
        self.curr_color = self.color_off

    def open_cv_writer(self):
        # Get video params and open opencv writer
        editor = Editor()
        self.cap = cv2.VideoCapture(self.videopath)
        if not self.cap.isOpened(): 
            return
            # raise FileNotFoundError(video)
        nframes, self.width, self.height, self.fps = editor.get_video_params(self.cap)

        self.writer = editor.open_cvwriter(self.savepath, w=self.width+self.border_size*2,
                                      h=self.height+self.border_size*2, framerate=self.fps, iscolor=True)

    def prep_squares(self):
        n_squares = len(self.stimuli.keys())
        centers = np.linspace(20, self.width-20, n_squares)

        complete_centers = [(int(c), int(self.height - (self.height*.95))) for c in centers]

        radius = int(np.diff(centers)[0]/3)
        if radius > 20: radius = 20
        return complete_centers, radius

    def make_concatenated_clips(self):
        pre_stim = self.fps*1
        post_stim = self.fps*11

        for stim_number, (stim_start, stim_dur) in enumerate(sorted(self.stimuli.values())):
            print('Adding new trial to the clip')
            # Get start and stop frames
            clip_start = stim_start-pre_stim
            clip_end = stim_start + (stim_dur*self.fps) + post_stim
            clip_number_of_frames = int(clip_end - clip_start)
            window = (clip_start, clip_end)
            
            # Move to n seconds before the start of the stimulus
            _ = self.get_selected_frame(self.cap, clip_start-1)

            # Keep reading frames until within post stim
            for frame_counter in tqdm(range(clip_number_of_frames)):
                ret, frame = self.cap.read()
                if not ret: break

                if not self.clean_vids:
                    frame_number = clip_start + frame_counter
                    if frame_number < window[0]+pre_stim or frame_number > window[1]-post_stim:
                        sign = ''
                        self.curr_color = self.color_off
                    else:
                        sign = '+'
                        self.curr_color = self.color_on
                        
                    frame = cv2.copyMakeBorder(frame, self.border_size, self.border_size, self.border_size, self.border_size,
                                                cv2.BORDER_CONSTANT, value=self.curr_color)
                    # cv2.circle(frame, (width-200, height-200), 30, curr_color, -1)
                    frame_time = (frame_number - window[0]) / self.fps
                    frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                    cv2.putText(frame, sign + str(frame_time) + 's', (self.width - 120, self.height - 100), 0, 1,
                                (20, 255, 20), thickness=2)

               
                    # Add circle to mark frame number
                    centers, radius = self.prep_squares()

                    for i, center in enumerate(centers):
                        if i == stim_number:
                            color = [200, 50, 50]
                            border = -1
                        else:
                            color = [255, 200, 200]
                            border = 5

                        cv2.circle(frame, (center[0], center[1]), radius, color, border)

                # Save to file
                self.writer.write(frame)


def create_trials_clips(prestim=3, poststim=25, clean_vids=True, plt_pose=False, all_trials=True, recs_range=None):
    def write_clip(video, savename, stim_frame, stim_duration, prestim, poststim, clean_vids, posedata):
        # parameters to draw on frame
        border_size = 0 # 20
        color_on = [100, 255, 100]
        color_off = [20,20,20]
        curr_color = color_off

        # Get video params and open opencv writer
        editor = Editor()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened(): 
            return
            # raise FileNotFoundError(video)
        nframes, width, height, fps = editor.get_video_params(cap)

        writer = editor.open_cvwriter(savename, w=width+border_size*2,
                                      h=height+border_size*2, framerate=fps, iscolor=True)

        # Get start and stop frames
        start = stim_frame - prestim*fps
        stop = stim_frame + poststim*fps
        clip_number_of_frames = int(stop-start)

        # Get stimulus window
        window = (prestim*fps, prestim*fps + stim_duration*fps)

        # Set cap to correct frame number
        real_start_frame = stim_frame - int(prestim*fps)
        cap.set(1, real_start_frame)

        # Write clip
        for frame_counter in tqdm(range(clip_number_of_frames)):
            ret, frame = cap.read()
            if not ret:
                if abs(frame_counter + start - nframes)<2:  # we are at the end of the clip
                    break
                else:
                    raise ValueError('Something went wrong when opening next frame: {} of {}'.
                                        format(frame_counter, nframes))

            # Overylay bodypart position of frame
            if posedata is not None and not clean_vids:
                real_frame_number = real_start_frame + frame_counter
                frame_pose = posedata.iloc[real_frame_number]
                points_dict = get_bps_as_points_dict(frame_pose)
                frame = cv2_plot_mouse_bps(frame, points_dict, s=2)

            # Prep to display stim on
            if frame_counter < window[0] or frame_counter > window[1]:
                sign = ''
                curr_color = color_off
            else:
                sign = '+'
                curr_color = color_on
                
            # Make frame
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = frame
            if not clean_vids:
                # gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size,
                #                             cv2.BORDER_CONSTANT, value=curr_color)
                # cv2.circle(gray, (width-200, height-200), 30, curr_color, -1)
                cv2.circle(gray, (width-100, height-50), 30, curr_color, -1)
                frame_time = (frame_counter - window[0]) / fps
                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                # cv2.putText(gray, sign + str(frame_time) + 's', (width - 250, height - 130), 0, 1,
                #             (20, 20, 20), thickness=2)

            # frame[:, :, 0] = gray

            # Save to file
            writer.write(gray)
        writer.release()

    # Get path to folder and list of previously saved videos
    paths = load_yaml('./paths.yml')
    # save_fld = os.path.join(paths['raw_data_folder'], paths['trials_clips'])
    save_fld = paths['trials_clips']
    saved_clips = [f for f in os.listdir(save_fld)]


    # Start looping over Recordings()
    recs = Recordings()
    behav_stims = BehaviourStimuli()
    mantis_stims = MantisStimuli()
    videofiles = VideoFiles()
    
    videos_df = pd.DataFrame(videofiles.fetch())
    videoname = None
    for recn, rec in enumerate(recs.fetch(as_dict=True)):
        # Get the stim table entry and clip ame
        # if rec['uid']<268: 
        #     continue

        if not recs_range[0] <= rec['uid'] <= recs_range[1]: continue

        print('Processing recording {} of {}'.format(recn, len(recs.fetch())))

        if rec['software'] == 'behaviour':
            stims = pd.DataFrame([s for s in behav_stims if s['recording_uid']==rec['recording_uid']])
            # raise NotImplementedError
        else:
            stims = pd.DataFrame([s for s in mantis_stims if s['recording_uid'] == rec['recording_uid']]).sort_values("overview_frame")
        stimuli_dict = {}
        for stimn, stim in stims.iterrows():
            print('     stim {} of {}'.format(stimn, len(stims)))
            clip_name = stim['stimulus_uid']+'.mp4'
            if clip_name in saved_clips: continue  # avoid doing again an old clip
                
            # Get frame time and video name for behaviour
            if rec['software'] == 'behaviour':
                videoname = stim['video']

                if plt_pose:
                    # Get pose data
                    videoentry = [v for v in videofiles if v['video_filepath']==videoname or v['converted_filepath'] == videoname][0]

                    posefile = videoentry['pose_filepath']
                    try:
                        posedata = pd.read_hdf(posefile)
                    except:
                        raise FileNotFoundError(posefile)
                else:
                    posedata = None

                # Write clip
                write_clip(videoname, os.path.join(save_fld, clip_name),
                            stim['stim_start'], stim['stim_duration'], 
                            prestim, poststim, clean_vids, posedata)

            else:
                dur = stim['duration']  # ! hardcoded duration in fps
                stimuli_dict[stimn] = (int(stim['overview_frame']), dur)

                # Get the corrisponding videofile
                # Get video path
                entry = videos_df.loc[(videos_df['recording_uid'] == stim['recording_uid']) & (videos_df['camera_name'] == 'overview')]
                videoname = entry['converted_filepath'].values[0]
                fld, name = os.path.split(videoname)

                correct_name =  name.split('__')[0]  # ! only necessary until database entry fixed
                clip_name = name.split('.')[0]+'_{}.mp4'.format(stimn)
                
                if os.path.exists(os.path.join(save_fld, clip_name)): 
                    print(clip_name, ' already exists')
                    continue

                print('Saving : ', os.path.join(save_fld, clip_name))


                write_clip(os.path.join(fld, correct_name), os.path.join(save_fld, clip_name),
                            int(stim['overview_frame']), dur, 
                            prestim, poststim, clean_vids, None)

                stimuli_dict[stimn] = (int(stim['overview_frame']), dur)

        if videoname is not None and all_trials:
            ClipWriter(videoname, stimuli_dict, clean_vids)



def parallelise_create_trials_clips():
    start_rec = 0
    end_rec = 300
    n_processes = 4

    arr = [x for x in np.arange(start_rec, end_rec)]
    splitted = np.array_split(arr, n_processes)
    limits = [(int(x[0]), int(x[-1])) for x in splitted]
    print(limits)

    # prestim=3, poststim=25, clean_vids=True, plt_pose=False, all_trials=True, recs_range=None
    params = (3, 25, False, False, False)
    partial_writer = partial(create_trials_clips, params)
    pool = ThreadPool(n_processes)
    _ = pool.map(partial_writer, limits)


def make_video_with_all_escapes(select_escapes=True, select_stim_evoked=True, select_exp = None, align_to_stim=True):
    savename = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\videos\\{}.mp4'.format(select_exp)
    # Get background
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))

    # Set options
    if align_to_stim:
        select_stim_evoked = True

    # Get returns data and tracking data
    all_escapes = pd.DataFrame(AllTrips().fetch())
    if select_escapes:
        all_escapes = all_escapes.loc[all_escapes['is_escape'] == 'true']
    else:
        is_this_an_escape = list(all_escapes['is_escape'].values)

    if select_stim_evoked:
        all_escapes = all_escapes.loc[all_escapes['is_trial'] == 'true']
    else:
        is_this_evoked = list(all_escapes['is_trial'].values)

    if select_exp:
        sq = ['Square Maze', 'TwoAndahalf Maze']
        if select_exp in sq:
            all_escapes = pd.concat([all_escapes.loc[all_escapes['experiment_name']==exp] for exp in sq])
        else:
            all_escapes = all_escapes.loc[all_escapes['experiment_name']==select_exp]


    # Make random colors
    random_colors = [(int(np.random.randint(10, 240, 1)[0]), 
                        int(np.random.randint(10, 240, 1)[0]),
                        int(np.random.randint(10, 240, 1)[0]))
                        for i in np.arange(1000)]

    # Make colors based on experiment
    experiments = list(all_escapes['experiment_name'].values)
    experiment_names = set(experiments)
    experiment_colors = {exp:(int(np.random.randint(10, 240, 1)[0]), 
                        int(np.random.randint(10, 240, 1)[0]),
                        int(np.random.randint(10, 240, 1)[0])) for exp in experiment_names}

    # Make colors based on order of arrival
    durations = list(all_escapes['duration'].values)
    durations_sortidx = np.argsort(durations)
    colorspace = np.linspace(10, 244, len(durations)+100)
    duration_colors = [(0, int(c/2), c) for c in colorspace]
    
    # make colors based on arm of origin and escape
    escape_arms = list(all_escapes['escape_arm'].values)
    arms = set(escape_arms)
    escape_arms_colors = {arm:((0, 255, 0) if 'Left' in arm else (255, 0, 0)) for arm in arms}

    # Open Video writer
    editor = Editor()
    writer = editor.open_cvwriter(savename, w=1000,
                                        h=1000, framerate=20, iscolor=True)

    # Extract tracking data
    bodyparts = ['left_ear', 'snout', 'right_ear', 'neck', 'body', 'tail_base']
    # bodyparts = ['snout', 'body']
    colors = dict(
        body=(255, 100, 100),
        snout=(100, 100, 255),
        left_ear = (75, 75, 200),
        right_ear = (75, 75, 200),
        neck = (200, 75, 75),
        tail_base = (150, 50, 50)
    )

    print('Ready to get data')
    tracking_datas = {bp:[] for bp in bodyparts}
    stim_onsets = []
    counter  = 0
    for index, row in all_escapes.iterrows():
        print('Fetching data from entry {} of {}'.format(counter, all_escapes.shape[0]))
        counter += 1
        if align_to_stim:
            t0, t1 = row['stim_frame']-30, row['shelter_enter']
        else:  # align to when the leave the threat platform
            t0, t1 = row['threat_exit']-90, row['shelter_enter']

        # Get the tracking data for each bodypart in each recording
        for bp in bodyparts:
            recording_tracking = get_tracking_given_recuid_and_bp(row['recording_uid'], bp)
            x = recording_tracking['tracking_data'].values[0][t0:t1, 0].astype(np.int16)
            y = recording_tracking['tracking_data'].values[0][t0:t1, 1].astype(np.int16)
            v =  line_smoother(recording_tracking['tracking_data'].values[0][t0:t1, 2].astype(np.int16))
            y = np.add(495, np.subtract(495, y))

            tracking_datas[bp].append(np.array([x, y, v]).T.astype(np.int16))

        stim_onsets.append(row['stim_frame'])

    # make videos 
    print('Ready to write video')
    framen = 0
    while True:
        # try:
        print('Writing frame: ', framen)
        bg = maze_model.copy()
        for bp in bodyparts:
            tracking = tracking_datas[bp]
            for i, tr in enumerate(tracking):
                if framen < tr.shape[0]:
                    max_v = int(tracking_datas['body'][i][framen, 2]/2)
                    # max_v = int(np.max(tracking_datas['body'][i][:, 2])/2)
                    if max_v < 2: max_v = 2
                    elif max_v > 12: max_v = 10

                    if 'snout' in bp.lower():
                        color= (0, 0, 255)
                    else:
                        color = escape_arms_colors[escape_arms[i]]

                    if select_escapes and not select_stim_evoked:
                        cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(max_v), color, -1)

                    elif not select_escapes and not select_stim_evoked:
                        if is_this_an_escape[i] == 'true':
                            cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(max_v),color, -1)
                        else:
                            cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(max_v), color, 1)

                    elif select_escapes and select_stim_evoked:
                            cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(max_v), color, -1)

                    else:
                        cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(max_v), color, -1)

        writer.write(bg)
        framen += 1
        if framen == 10*30: break

        # cv2.imshow('a', bg)
        # cv2.waitKey(1)
    writer.release()


def make_video_with_all_explorations():
    savename = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\all_explorations.mp4'
    # Get background
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))

    # Get returns data and tracking data
    all_explorations = pd.DataFrame(AllExplorations().fetch())

    # Open Video writer
    editor = Editor()
    writer = editor.open_cvwriter(savename, w=1000,
                                        h=1000, framerate=120, iscolor=True)

    # Extract tracking data
    bodyparts = ['left_ear', 'snout', 'right_ear', 'neck', 'body', 'tail_base']
    # bodyparts = ['snout', 'body']
    colors = dict(
        body=(255, 100, 100),
        snout=(100, 100, 255),
        left_ear = (75, 75, 200),
        right_ear = (75, 75, 200),
        neck = (200, 75, 75),
        tail_base = (150, 50, 50)
    )

    print('Ready to get data')
    tracking_datas = {bp:[] for bp in bodyparts}
    counter  = 0
    for index, row in all_explorations.iterrows():
        print('Fetching data from entry {} of {}'.format(counter, all_explorations.shape[0]))
        counter += 1

        # Get the tracking data for each bodypart in each recording
        for bp in bodyparts:
            # recording_tracking = get_tracking_given_recuid_and_bp(row['recording_uid'], bp)
            # x = recording_tracking['tracking_data'].values[0][:, 0].astype(np.int16)
            # y = recording_tracking['tracking_data'].values[0][:, 1].astype(np.int16)
            # v = recording_tracking['tracking_data'].values[0][:, 2].astype(np.int16)
            x = row['tracking_data'][:, 0].astype(np.int16)
            y = row['tracking_data'][:, 1].astype(np.int16)
            v = row['tracking_data'][:, 2].astype(np.int16)
            y = np.add(490, np.subtract(490, y))
            tracking_datas[bp].append(np.array([x, y, v]).T)

        # if counter == 10: break

    # Calc video duration
    durations = all_explorations['duration'].values
    number_of_frames = np.percentile(durations, 75)*30

    # make videos 
    print('Ready to write video')
    for framen in np.arange(int(number_of_frames)):
        print('Writing frame: ', framen, ' of ', number_of_frames)
        bg = maze_model.copy()
        for bp in bodyparts:
            tracking = tracking_datas[bp]
            for i, tr in enumerate(tracking):
                if framen < tr.shape[0]:
                    v = tr[framen, -1]
                    if v < 2: v = 2
                    elif v > 15: v = 15
                    cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), int(v), colors[bp], -1)

        writer.write(bg)
        framen += 1


        # cv2.imshow('a', bg)
        # cv2.waitKey(1)
    writer.release()



if __name__ == "__main__":
    # paths = load_yaml('./paths.yml')

    # create_trials_clips(clean_vids=False, all_trials=False, recs_range=[100, 120])
    # make_video_with_all_escapes(select_exp='PathInt2')

    # parallelise_create_trials_clips()

    tcm = TrialClipsMaker(process_recs_in_range = [276, 1000], 
                            add_threat_video    = True, 
                            overlay_pose        = False, 
                            overlay_text        = True)
    tcm.loop_over_recordings()

