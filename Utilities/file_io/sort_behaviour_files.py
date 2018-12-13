import sys
sys.path.append('./')  

import os
from shutil import copyfile

from Utilities.file_io.files_load_save import load_yaml

def sort_behaviour_files(tosort_fld, video_fld, metadata_fld):
    yn = input('WARNING! this function only works with Behaviour software, NOT Mantis.\n Continue? y/n       ')
    if 'y' not in yn.lower(): return
    for fld in os.listdir(tosort_fld):
        for f in os.listdir(os.path.join(tosort_fld, fld)):
            print('sorting ', fld)
            if '.tdms' in f:
                if 'index' in f: continue
                dst = os.path.join(metadata_fld, f)
                if f in os.listdir(metadata_fld): 
                    print('Already moved')
                    continue
                else:
                    copyfile(os.path.join(tosort_fld, fld, f), dst)
            elif '.mp4' in f:
                dst = os.path.join(video_fld, fld+'.mp4')
                if fld+'.mp4' in os.listdir(video_fld):
                    print('Already moved')
                    continue
                else:
                    os.rename(os.path.join(tosort_fld, fld, f),
                                os.path.join(tosort_fld, fld, fld+'.mp4'))

                    copyfile(os.path.join(tosort_fld, fld, fld+'.mp4'), dst)
            else:
                raise ValueError('Could not proess file with format: ', os.path.split(f)[-1])
    print('... task completed')


def sort_mantis_files():
    # Get folders paths
    paths = load_yaml('paths.yml')
    raw = paths['raw_data_folder']
    metadata_fld = os.path.join(raw, paths['raw_metadata_folder'])
    video_fld = os.path.join(raw, paths['raw_video_folder'])
    tosort_fld = os.path.join(raw, paths['raw_to_sort'])
    ai_fld = os.path.join(raw, paths['raw_analoginput_folder'])

    log = open(os.path.join(tosort_fld, 'log.txt'), 'w+')
    log.write('\n\n\n\n')
    # Loop over subfolders in tosort_fld
    for fld in os.listdir(tosort_fld):
        log.write('Processing Folder {}'.format(fld))
        print('Processing Folder ', fld)
        # Loop over individual files in subfolder
        if  '.txt' in fld: continue # skip log file
        for f in os.listdir(os.path.join(tosort_fld, fld)):
            if '.txt' in f: continue # skip log file
            print('     Moving: ', f)
            # Get the new name and destination for each file
            if 'Maze' in f:
                # Its the file with the AI
                newname = fld+'.tdms'
                dest = ai_fld
            elif 'Overview' in f:
                newname = fld+'Overview.tdms'
                if 'meta' in f:
                    # Overview Camera Metadata file
                    dest = metadata_fld
                else:
                    # Overview Camera Data file
                    dest = video_fld
            elif 'Threat' in f:
                newname = fld+'Threat.tdms'
                if 'meta' in f:
                    # Threat Camera Metadata file
                    dest = metadata_fld
                else:
                    # Threat Camera Data file
                    dest = video_fld
            else:
                raise ValueError('Unexpected file: ', os.path.join(fld, f))

            original = os.path.join(tosort_fld, fld, f)
            moved = os.path.join(dest, newname)
            try:
                os.rename(original, moved)
                log.write('Moved {} to {}'.format(original, moved))
            except:
                print('         Didnt move file because already exists')
                log.write('!!NOT!!! Moved {} to {}'.format(original, moved))
        log.write('Completed Folder {}\n\n'.format(fld))
    log.close()

if __name__ == "__main__":
    sort_mantis_files()

