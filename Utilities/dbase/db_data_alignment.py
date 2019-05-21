import sys
sys.path.append('./')  

from Utilities.imports import *


from Utilities.video_and_plotting.video_editing import *
from Utilities.maths.filtering import butter_lowpass_filter
from Utilities.maths.stimuli_detection import find_peaks_in_signal
from database.database_fetch import *
from Utilities.file_io.files_load_save import *


class ThreatDataProcessing:
    def __init__(self, table, key, test_mode=False):
        print("Processing rec: ", key['recording_uid'])
        self.feathers_folder = "Z:\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata\\as_pandas"

        self.set_params()

        self.key = key
        self.table = table
        self.test_mode = test_mode

        self.overview_ch = "/'OverviewCameraTrigger_AI'/'0'"
        self.threat_ch = "/'ThreatCameraTrigger_AI'/'0'"
        self.frame_times = {}


        if test_mode:
            from database.TablesPopulateFuncs import ToolBox

            self.tool_box = ToolBox()

            self.test_folder = "Z:\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata\\as_pandas"
            self.test_file = "190513_CA601_1.ft"
            # self.make_feathers()
            self.load_a_feather()

        else:
            found = self.fetch_files()
            return found

    def set_params(self):
        self.sampling_rate = 25000
        self.filter_cutoff = 5000
        self.peaks_min_iti = 6
        self.peaks_th = 4

        self.start_time = int( 0.2 * 10**8)  # ? when testing don't process whole data, make things faster
        self.end_time = int(0.4 * 10**8)
    
    def fetch_files(self):
        # get video file for recording
        videos = get_videos_given_recuid(self.key["recording_uid"])
        if len(videos) == 1:
            print("     only one video for this rec, skipping")
            self.feather_file = None
        else:
            aifilepath = get_rec_aifilepath_given_recuid(self.key['recording_uid'])[0]

            # look for a feather file
            fld, file_name = os.path.split(aifilepath)

            feathers = os.listdir(os.path.join(fld, "as_pandas"))

            feather = [f for f in feathers if file_name.split(".")[0] in f]
            if feather:
                self.feather_file = os.path.join(fld, "as_pandas", feather[0])
            else:
                print("     no feather found")
                self.feather_file = None


    def make_feathers(self):
        for f in os.listdir(self.test_folder)[::-1]:
            if ".tdms" in f:
                feather_name = f.split(".")[0]+".ft"
                if not feather_name in os.listdir(self.test_folder):
                    content = self.tool_box.open_temp_tdms_as_df(os.path.join(self.test_folder, f), move=False, skip_df=False, memmap_dir=self.test_folder )
                    print("         ... saving")
                    content[0].to_feather(os.path.join(self.test_folder, feather_name))
                    print("                ... saved")

    def load_a_feather(self):
        if self.test_mode:
            print("\n\nLoading: ", self.test_file)
            self.data = load_feather(os.path.join(self.test_folder, self.test_file))[self.start_time: self.end_time]
        else:
            print("     loading data")
            self.data = load_feather(self.feather_file)

    def process_channel(self, ch, key):
        # ? We need to filter because sometimes there is quite a lot of high freq noise
        # and this gets picked up as a frame otherwise
        filtered_signal = butter_lowpass_filter(self.data[ch].values, self.filter_cutoff, self.sampling_rate)
        self.frame_times[key] = np.add(find_peaks_in_signal(filtered_signal, self.peaks_min_iti, self.peaks_th), self.start_time)

    def test_filter(self):
        f, ax = plt.subplots()
        filtered1 = butter_lowpass_filter(self.data[self.threat_ch], 6000, 25000)
        filtered2 = butter_lowpass_filter(self.data[self.threat_ch], 10000, 25000)

        ax.plot(self.data[self.threat_ch].values, color='k', linewidth=3, alpha=1)
        ax.plot(filtered1, color='r', linewidth=2, alpha=.5)
        ax.plot(filtered2, color='g', linewidth=2, alpha=.5)

    def plot_channels(self):
        f, ax = plt.subplots()
        ax.plot(self.data[self.overview_ch], color='c', linewidth=3)
        ax.plot(self.data[self.threat_ch], color='r', linewidth=1)

        ax.plot(self.frame_times["threat"], [-.4 for x in self.frame_times["threat"]], "o", color="r")
        ax.plot(self.frame_times["overview"], [-.8 for x in self.frame_times["overview"]], "o", color="c")

    def align_frames(self):
        # Create an array with 3 columns -> frame IDX, Overview Frame timestamp, zeros
        aligned_frames = np.vstack([np.arange(len(self.frame_times['overview'])), self.frame_times['overview'], np.zeros_like(self.frame_times['overview'])]).T.astype(np.float32)

        if np.any(self.frame_times['threat']):
            # avg time between threat frames
            avg_IFI = np.mean(np.diff(aligned_frames[:, 1]))

            # for each overview frame match the closest threat frame
            for i in np.arange(aligned_frames.shape[0]):
                closest_threat_frame = self.frame_times['threat'][np.argmin(self.frame_times['threat']-aligned_frames[i, 1])]
                delta = abs(closest_threat_frame - aligned_frames[i, 1])
                if delta > avg_IFI:
                    aligned_frames[i, -1] = np.nan
                else:
                    aligned_frames[i, -1] = closest_threat_frame

        self.frame_times['aligned'] = aligned_frames
            

    def insert_in_table(self):
        # add stuff to key
        self.key['overview_frames_timestamps']  = self.frame_times['overview']
        self.key['threat_frames_timestamps']    = self.frame_times['threat']
        self.key['aligned_frame_timestamps']    = self.frame_times['aligned']

        self.table.insert1(self.key)



if __name__ == "__main__":
    tdp = ThreatDataProcessing(test_mode = True)

    tdp.process_channel(tdp.threat_ch, "threat")
    tdp.process_channel(tdp.overview_ch, "overview")

    # tdp.plot_channels()
    # tdp.test_filter()

    tdp.align_frames()

    plt.show()



