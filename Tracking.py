import datajoint as dj
import os
from pathlib import Path
import pyexcel
try: import yaml
except: pass
from nptdms import TdmsFile

from Tables_definitions import *



dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'


dj.conn()


class Track:
    """
    Use DLC to track for one, many or all BehaviourRecordings in the database

    """

    def __init__(self, recordings=None):
        self.recordings = recordings

        self.sessions_table = Session()
        self.mice_table = Mouse()
        self.recordings_table = BehaviourRecording()

        if recordings is None:
            run_all = True
        else:
            run_all = False

        self.run_tracking(run_all)

    def run_tracking(self, run_all):
        if run_all:
            all_recordings = self.recordings_table.fetch(as_dict=True)

            for rec in all_recordings:
                stim_times = self.get_stim_frames(rec['metadata_path'])

                a = 1

    @staticmethod
    def get_stim_frames(f):
        print('           ... loading metadata from .tdms: {}'.format(os.path.split(f)[-1]))
        tdms = TdmsFile(f)

        stimuli = {s:None for s in ['visual', 'audio']}
        visual_rec_stims, audio_rec_stims = [], []
        for group in tdms.groups():
            for obj in tdms.group_channels(group):
                if 'stimulis' in str(obj).lower():
                    for idx in obj.as_dataframe().loc[0].index:
                        if '  ' in idx:
                            framen = int(idx.split('  ')[1].split('-')[0])
                        else:
                            framen = int(idx.split(' ')[2].split('-')[0])
                        if 'visual' in str(obj).lower():
                            visual_rec_stims.append(framen)
                        elif 'audio' in str(obj).lower():
                            audio_rec_stims.append(framen)

                        else:
                            print('                  ... couldnt load stim correctly')

        stimuli['visual'] = visual_rec_stims
        stimuli['audio'] = audio_rec_stims
        return stimuli



class TrackingData:
    def __init__(self):
        self.stimuli = None
        self.bodyparts = None

        self.dlc_data = {}

        self.tracking_data = {}







if __name__ == '__main__':
    Track()





