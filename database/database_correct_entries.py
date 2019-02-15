import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project

# from database.dj_config import start_connection
# start_connection()
from database.NewTablesDefinitions import *
from database.auxillary_tables import *

import pandas as pd

def fix_videofiles():
    """
        Detect entries in VideoFiles that have wrong paths and correct them

    """

    videofiles = pd.DataFrame(VideoFiles.fetch())

    recordings_to_delete = []

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

        if 'joined' in row['converted_filepath']:
            if not os.path.isfile(row['converted_filepath']):
                name, ext = row['converted_filepath'].split('.')
                name, ext = name.split('__')[0], '.'+ext
                cleaned_name = name+ext

                if os.path.isfile(cleaned_name):
                    print('Good: ', cleaned_name)
                    recordings_to_delete.append(row['uid'])
                else:
                    if 'Threat' in row['converted_filepath']: continue
                    raise FileNotFoundError('Dang')

                # # Delete entry
                # (VideoFiles & "recording_uid={}".format(row['recording_uid'])).delete()

    all_tracking_uids = TrackingData.fetch("uid")
    all_body_uids = TrackingDataJustBody.fetch("uid")

    # Clean up tables
    for r in recordings_to_delete:
        if r in all_tracking_uids:
            (TrackingData & "uid={}".format(r)).delete()

        if r in all_body_uids:
            (TrackingDataJustBody & "uid={}".format(r)).delete()


        (VideoFiles & "uid={}".format(r)).delete()
            


    a = 1



if __name__ == "__main__":
    fix_videofiles()
