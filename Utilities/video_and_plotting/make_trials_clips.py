import sys
sys.path.append('./') 

import os 
import cv2
from tqdm import tqdm
import pandas as pd

from Utilities.file_io.files_load_save import load_yaml
from Utilities.video_and_plotting.video_editing import Editor
from database.NewTablesDefinitions import *
from Processing.plot.plotting_utils import *
from Processing.plot.video_plotting_toolbox import *


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

    @staticmethod
    def get_selected_frame(cap, show_frame):
            cap.set(1, show_frame)
            ret, frame = cap.read() # read the first frame
            return frame

    def prep_squares(self):
        n_squares = len(self.stimuli.keys())
        centers = np.linspace(20, self.width-20, n_squares-1)

        complete_centers = [(int(c), int(self.height - (self.height*.95))) for c in centers]

        radius = int(np.diff(centers)[0]/3)
        if radius > 20: radius = 20
        return complete_centers, radius


    def make_concatenated_clips(self):
        pre_stim = self.fps*1
        post_stim = self.fps*2

        for stim_number, (stim_start, stim_dur) in enumerate(self.stimuli.values()):
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




def create_trials_clips(prestim=10, poststim=10, clean_vids=True, plt_pose=False):
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
                cv2.circle(gray, (width-200, height-200), 30, curr_color, -1)
                frame_time = (frame_counter - window[0]) / fps
                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                cv2.putText(gray, sign + str(frame_time) + 's', (width - 250, height - 130), 0, 1,
                            (20, 20, 20), thickness=2)

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

    for recn, rec in enumerate(recs.fetch(as_dict=True)):
        # Get the stim table entry and clip ame
        print('Processing recording {} of {}'.format(recn, len(recs.fetch())))
        if rec['uid']<194: 
            print(' ... skipped')
            continue

        if rec['software'] == 'behaviour':
            stims = [s for s in behav_stims if s['recording_uid']==rec['recording_uid']]
        else:
            stims = [s for s in mantis_stims if s['recording_uid'] == rec['recording_uid']]


        

        stimuli_dict = {}
        for stimn, stim in enumerate(stims):
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

        ClipWriter(videoname, stimuli_dict, clean_vids)
                
                


def make_video_with_all_escapes():
    savename = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\all_trials4.mp4'
    # Get background
    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    # maze_model = cv2.cvtColor(maze_model,cv2.COLOR_RGB2GRAY)

    # Get returns data and tracking data
    all_escapes = pd.DataFrame(AllTrips().fetch())
    # all_escapes = all_escapes.loc[all_escapes['is_escape'] == 'true']
    is_this_an_escape = all_escapes['is_escape']


    # Open Video writer
    editor = Editor()
    writer = editor.open_cvwriter(savename, w=1000,
                                        h=1000, framerate=30, iscolor=True)

    # Extract tracking data
    print('Ready to get data')
    tracking_datas = []
    for index, row in all_escapes.iterrows():
        t0, t1 = row['threat_exit']-row['shelter_exit']-30, row['shelter_enter']-row['shelter_exit']
        tracking = row['tracking_data'][t0:t1].astype(np.int16)
        x= tracking[:, 0]
        y = np.add(490, np.subtract(490, tracking[:, 1]))  # Invert on the Y axis
        v = tracking[:, 2] 

        tracking_datas.append(np.array([x, y, v]).T)

    # make videos 
    print('Ready to write video')
    framen = 0
    while True:
        # try:
        print('Writing frame: ', framen)
        bg = maze_model.copy()
        for i, tr in enumerate(tracking_datas):
            if framen < tr.shape[0]:
                vv = np.int(tr[framen, 2])*50
                if vv > 255: vv=255
                if is_this_an_escape[i] == 'true':
                    color = (0, vv, 0)
                else:
                    color = (0, 0, vv)
                cv2.circle(bg, (tr[framen, 0], tr[framen, 1]), 3, color, -1)

        writer.write(bg)
        framen += 1
        if framen == 10*30: break

        # cv2.imshow('a', bg)
        # cv2.waitKey(1)
    writer.release()



if __name__ == "__main__":
    paths = load_yaml('./paths.yml')

    # create_trials_clips(clean_vids=False)
    make_video_with_all_escapes()

