import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

# from database.dj_config import start_connection
# start_connection()
from database.NewTablesDefinitions import *

import pandas as pd

def fix_videofiles():
    """
        Detect entries in VideoFiles that have wrong paths and correct them

    """

    videofiles = pd.DataFrame(VideoFiles.fetch())

    for idx, row in videofiles.iterrows():
        print("""
        Recording:      {}
        -------------
        camera:         {}
        video:          {}
        converted:      {}
        metadata:       {}
        pose:           {}
        \n\n
        """.format(row['recording_uid'], row['camera_name'], row['video_filepath'], row['converted_filepath'], row['metadata_filepath'], row['pose_filepath']))

        if '__' in row['converted_filepath']:
            if not os.path.isfile(row['converted_filepath']):
                name, ext = row['converted_filepath'].split('.')
                name, ext = name.split('__')[0], '.'+ext
                cleaned_name = name+ext

                if os.path.isfile(cleaned_name):
                    print('Good: ', cleaned_name)
                else:
                    if 'Threat' in row['converted_filepath']: continue
                    raise FileNotFoundError('Dang')

                # # Delete entry
                # (VideoFiles & "recording_uid={}".format(row['recording_uid'])).delete()




    a = 1



if __name__ == "__main__":
    fix_videofiles()
