import sys
sys.path.append('./')

from Utilities.imports import *

cur_dir = os.getcwd()
os.chdir("C:\\Users\\Federico\\Documents\\GitHub\\VisualStimuli")
from Utils.contrast_calculator import Calculator as ContrastCalc
os.chdir(cur_dir)

from nptdms import TdmsFile
import scipy.signal as signal
from collections import OrderedDict

from Utilities.video_and_plotting.commoncoordinatebehaviour import run as get_matrix
from Utilities.maths.stimuli_detection import *
from Utilities.dbase.stim_times_loader import *

from Processing.tracking_stats.correct_tracking import correct_tracking_data
from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame
from Processing.tracking_stats.extract_velocities_from_tracking import complete_bp_with_velocity, get_body_segment_stats

# from database.auxillary_tables import *



class ToolBox:
    def __init__(self):
        # Load paths to data folders
        self.paths = load_yaml('paths.yml')
        self.raw_video_folder = os.path.join(
            self.paths['raw_data_folder'], self.paths['raw_video_folder'])
        self.raw_metadata_folder = os.path.join(
            self.paths['raw_data_folder'], self.paths['raw_metadata_folder'])
        self.tracked_data_folder = self.paths['tracked_data_folder']
        self.analog_input_folder = os.path.join(self.paths['raw_data_folder'], 
                                                self.paths['raw_analoginput_folder'])
        self.pose_folder = self.paths['tracked_data_folder']

    def get_behaviour_recording_files(self, session):
        raw_video_folder = self.raw_video_folder
        raw_metadata_folder = self.raw_metadata_folder

        # get video and metadata files
        videos = sorted([f for f in os.listdir(raw_video_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f
                            and '.h5' not in f and '.pickle' not in f])
        metadatas = sorted([f for f in os.listdir(raw_metadata_folder)
                            if session['session_name'].lower() in f.lower() and 'test' not in f and '.tdms' in f])

        if videos is None or metadatas is None:
            raise FileNotFoundError(videos, metadatas)

        # Make sure we got the correct number of files, otherwise ask for user input
        if not videos or not metadatas:
            if not videos and not metadatas:
                return None, None
            raise FileNotFoundError('dang')
        else:
            if len(videos) != len(metadatas):
                # print('Found {} videos files: {}'.format(len(videos), videos))
                # print('Found {} metadatas files: {}'.format(
                #     len(metadatas), metadatas))
                raise ValueError(
                    'Something went wront wrong trying to get the files')

            num_recs = len(videos)
            # print(' ... found {} recs'.format(num_recs))
            return videos, metadatas


    def tdms_as_dataframe(self, tdms, to_keep, time_index=False, absolute_time=False):
        """
        Converts the TDMS file to a DataFrame

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """
        keys = []  # ? also return all the columns as well
        dataframe_dict = OrderedDict()
        for key, value in tdms.objects.items():
            keys.append(key)
            if key not in to_keep: continue
            if value.has_data:
                index = value.time_track(absolute_time) if time_index else None
                dataframe_dict[key] = pd.Series(data=value.data, index=index)
        return pd.DataFrame.from_dict(dataframe_dict), keys

    def open_temp_tdms_as_df(self, path, move=True, skip_df=False, memmap_dir = None):
        """open_temp_tdms_as_df [gets a file from winstore, opens it and returns the dataframe]
        
        Arguments:
            path {[str]} -- [path to a .tdms]
        """
        # Download .tdms from winstore, and open as a DataFrame
        # ? download from winstore first and then open, faster?
        if move:
            try:
                temp_file = load_tdms_from_winstore(path)
            except:
                raise ValueError("Could not move: ", path)
        else:
            temp_file = path

        print('opening ', temp_file, ' with size {} GB'.format(
            round(os.path.getsize(temp_file)/1000000000, 2)))
        bfile = open(temp_file, 'rb')
        print("  ... opened binary, now open as TDMS")

        if memmap_dir is None: memmap_dir = "M:\\"
        tdmsfile = TdmsFile(bfile, memmap_dir=memmap_dir)
        print('      ... TDMS opened')
        if skip_df:
            return tdmsfile, None
        else:
            print("          ... opening as dataframe")
            groups_to_keep = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'LDR_signal_AI'/'0'", "/'AudioIRLED_analog'/'0'", "/'WAVplayer'/'0'"]
            tdms_df, cols = self.tdms_as_dataframe(tdmsfile, groups_to_keep)
            print('              ... opened as dataframe')

            return tdms_df, cols


    def extract_behaviour_stimuli(self, aifile):
        """extract_behaviour_stimuli [given the path to a .tdms file with session metadata extract
        stim names and timestamp (in frames)]
        
        Arguments:
            aifile {[str]} -- [path to .tdms file] 
        """
        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=False)

        stim_cols = [c for c in cols if 'Stimulis' in c]
        stimuli = []
        stim = namedtuple('stim', 'type name frame')
        for c in stim_cols:
            stim_type = c.split(' Stimulis')[0][2:].lower()
            if 'digit' in stim_type: continue
            stim_name = c.split('-')[-1][:-2].lower()
            try:
                stim_frame = int(c.split("'/' ")[-1].split('-')[0])
            except:
                try:
                    stim_frame = int(c.split("'/'")[-1].split('-')[0])
                except:
                    continue
            stimuli.append(stim(stim_type, stim_name, stim_frame))
        return stimuli

    def extract_ai_info(self, key, aifile):
        """
        aifile: str path to ai.tdms

        extract channels values from file and returns a key dict for dj table insertion

        """

        # Get .tdms as a dataframe
        tdms_df, cols = self.open_temp_tdms_as_df(aifile, move=True, skip_df=True)
        chs = ["/'OverviewCameraTrigger_AI'/'0'", "/'ThreatCameraTrigger_AI'/'0'", "/'AudioIRLED_AI'/'0'", "/'AudioFromSpeaker_AI'/'0'"]
        """ 
        Now extracting the data directly from the .tdms without conversion to df
        """
        key['overview_camera_triggers'] = np.round(tdms_df.object('OverviewCameraTrigger_AI', '0').data, 2)
        key['threat_camera_triggers'] = np.round(tdms_df.object('ThreatCameraTrigger_AI', '0').data, 2)
        key['audio_irled'] = np.round(tdms_df.object('AudioIRLED_AI', '0').data, 2)
        if 'AudioFromSpeaker_AI' in tdms_df.groups():
            key['audio_signal'] = np.round(tdms_df.object('AudioFromSpeaker_AI', '0').data, 2)
        else:
            key['audio_signal'] = -1
        key['ldr'] = -1  # ? insert here
        key['tstart'] = -1
        key['manuals_names'] = -1
        # warnings.warn('List of strings not currently supported, cant insert manuals names')
        key['manuals_timestamps'] = -1 #  np.array(times)
        return key