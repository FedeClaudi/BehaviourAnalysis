import pandas as pd
import os
import yaml
import sys
sys.path.append('./') 

try:
    os.chdir("C:\\GITHUB\\BehaviourAnalysis")
    videofolder = "W:\\branco\\Federico\\raw_behaviour\\maze\\video"
except:
    videofolder = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'

try:
    from database.Populate_database import PopulateDatabase
    from database.NewTablesDefinitions import *
    from database.auxillary_tables import VideoTdmsMetadata
except:
    pass

import time

from Utilities.video_and_plotting.video_editing import VideoConverter, Editor
from Utilities.file_io.sort_behaviour_files import sort_mantis_files
from database.TablesPopulateFuncs import ToolBox
from Utilities.file_io.files_load_save import *


""" [Toolbox functions to automate handling of files (e.g. convert videos from .tdms to .avi...)]
"""

class FilesAutomationToolbox:
    def __init__(self):
        self.tool_box = ToolBox()

        # self.database = PopulateDatabase()
        paths = load_yaml("paths.yml")
        self.videos_fld = videofolder
        self.pose_fld = paths['tracked_data_folder']
        self.ai_fld = os.path.join(paths['raw_data_folder'], paths['raw_analoginput_folder'])
        self.ai_dest_fld = os.path.join(self.ai_fld, "as_pandas")

        try:
            self.video_metadata = VideoTdmsMetadata
        except:
            self.video_metadata = None

    def save_ai_files_as_pandas(self):
        files = [f for f in os.listdir(self.ai_fld) if '.yml' not in f and "." in f and ".tdms" in f]
        for i, ai in enumerate(files):
            print("\n\nProcessing file {} of {}".format(i,len(files)))

            date = int(ai.split("_")[0])
            if date < 190509: continue  # when started with visuals

            savename = ai.split('.')[0]+".ft"
            columns_savename = ai.split('.')[0]+"_groups.yml"
            if savename in os.listdir(self.ai_dest_fld): continue

            try:
                content = self.tool_box.open_temp_tdms_as_df(os.path.join(self.ai_fld, ai), move=True, skip_df=False)
            except:
                raise ValueError("Could not open: ", ai)

            print("         ... saving")
            content[0].to_feather(os.path.join(self.ai_dest_fld, savename))
            save_yaml(os.path.join(self.ai_dest_fld, columns_savename), content[1])
            print("                ... saved")

    def extract_videotdms_metadata(self):
        """[Populate a dj table with the videos metadata]
        """

        table = VideoTdmsMetadata()
        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        for t in tdmss:
            props, number_of_frames = VideoConverter.extract_framesize_from_metadata(os.path.join(self.videos_fld, t))
            key = dict(videopath=os.path.join(self.videos_fld, t),
                        width=props['width'],
                        height=props['height'],
                        number_of_frames=number_of_frames,
                        fps=props['fps'])
            try:
                table.insert1(key)
            except:
                pass
                # raise ValueError('Could not insert: ', key)
        print(table)

    def convert_tdms_to_mp4(self, n_processes=1):
        """
            Keeps calling video conversion tool, regardless of what happens
        """
        try:
            sort_mantis_files()
        except:
            pass

        tcvt = self.get_list_uncoverted_tdms_videos()
        toconvert = [os.path.join(self.videos_fld, t) for t in tcvt]
        # while True:
        if n_processes> 1:
            in_process = None
            try:
                for f in toconvert:
                    in_process = f
                    VideoConverter(os.path.join(self.videos_fld, f), extract_framesize=True)
                print('All files converted, yay!')

                editor = Editor()
                editor.concated_tdms_to_mp4_clips(fld)
                print('All clips joined, yay!')

                # break
            except: # ignore exception and try again
                if in_process is not None:
                    pass # ! clear unfinished business 
                print('Failed again, trying again..\n\n')
        else:
            for f in toconvert:
                try:
                    VideoConverter(os.path.join(self.videos_fld, f), extract_framesize=True)
                except:
                    continue

    @staticmethod
    def check_if_file_converted(name, folder):
        conv, join = False, False
        mp4s = [v for v in os.listdir(folder) if name in v and '.mp4' in v]
        # print(name)
        if mp4s:
            joined = [v for v in mp4s if 'joined' in mp4s]
            conv = True
            if joined: join=True

        return conv, join

    def get_list_uncoverted_tdms_videos(self):
        """
            Check which videos still need to be converted
        """
        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        unconverted = []
        for t in tdmss:
            name = t.split('.')[0]
            conv, join = self.check_if_file_converted(name, self.videos_fld)
            if not conv: unconverted.append(t)
            
        # store names to file
        store = "Utilities/file_io/files_to_convert.yml"
        with open(store, 'w') as out:


            yaml.dump([os.path.split(u)[-1] for u in unconverted], out)


        print('To convert: ', unconverted)
        print(len(unconverted), ' files yet to convert')
        return unconverted

    def get_list_not_tracked_videos(self):
        videos = [f.split('.')[0] for f in os.listdir(self.videos_fld) if 'tdms' not in f and "." in f]
        poses = [f.split('_')[:-1] for f in os.listdir(self.pose_fld) if 'h5' in f]

        not_tracked = [f for f in videos if f.split('_') not in poses] #  and 'overview'  in f.lower()]
        # not_tracked = [f for f in videos if f.split('_') not in poses]
        # remove threat videos that are old and dont want to track
        not_tracked = [f for f in not_tracked if int(f.split("_")[0]) > 190500]

        print('To track: ', not_tracked)
        print(len(not_tracked), ' files yet to track')

        store = "Utilities/file_io/files_to_track.yml"
        with open(store, 'w') as out:
            yaml.dump([os.path.split(u)[-1]+".mp4" for u in not_tracked], out)

        return not_tracked

    def check_video_conversion_correct(self):
        """[Check if the video conversion was done correctly, i.e. we have all the frames and all clips have same length]
        
        Raises:
            ValueError -- [If something went wrong raise an error]
        """

        tdmss = [f for f in os.listdir(self.videos_fld) if '.tdms' in f]
        editor = Editor()

        for t in tdmss:
            print('\n\nChecking: ', t)
            name = t.split('.')[0]
            conv, join = self.check_if_file_converted(name, self.videos_fld)
            number_of_frames = []
            if conv and not join:
                mp4s = [v for v in os.listdir(self.videos_fld) if name in v and '.mp4' in v]
                for mp4 in mp4s:
                    if [f for f in ['top', 'side', 'catwalk'] if f in mp4]: continue # ignore cropped videos
                    cap = cv2.VideoCapture(os.path.join(self.videos_fld, mp4))
                    nframes, width, height, fps = editor.get_video_params(cap)
                    number_of_frames.append(nframes)
                if not number_of_frames or number_of_frames[0] == 0:
                    continue 

                # Check that we are sure about the total number of frames
                tot = np.sum(np.array(number_of_frames))
                metadata = pd.DataFrame(self.video_metadata.fetch())
                entry = metadata.loc[metadata['videopath'] == os.path.join(self.videos_fld, t)]
                if not entry.shape[0]: raise FileNotFoundError('Load video metadata first')

                if tot != entry['number_of_frames'].values[0]: raise ValueError('Tot frames, converted frames:', a, number_of_frames)

                else: 
                    print(' ...  correctly converted all ',  tot, 'frames [{} converted]'.format(tot))
                # else: 
                #     raise ValueError('Tot frames, converted frames:', a, number_of_frames)

    def remove_stupid_videofiles(self):
        dr = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\video'
        for f in os.listdir(dr):
            if 'top' in f or 'side' in f or 'catwalk' in f:
                if os.path.getsize(os.path.join(dr, f)) < 2000:
                    os.remove(os.path.join(dr, f))
 



if __name__ == "__main__":
    automation = FilesAutomationToolbox()

    # automation.convert_tdms_to_mp4()

    automation.get_list_uncoverted_tdms_videos()
    automation.get_list_not_tracked_videos()

    # Checks 
    # automation.extract_videotdms_metadata()
    # automation.check_video_conversion_correct() 
    # automation.remove_stupid_videofiles()


    # automation.save_ai_files_as_pandas()
# 
