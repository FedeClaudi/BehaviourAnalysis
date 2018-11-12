import os    
    
def sort_behaviour_files(tosort_fld, video_fld, metadata_fld):
    yn = input('WARNING! this function only works with Behaviour software, NOT Mantis.\n Continue? y/n')
    if 'y' not in yn.lower(): return
    for fld in os.listdir(tosort_fld):
        for f in os.listdir(os.path.join(tosort_fld, fld)):
            print('sorting ', fld)
            if '.tdms' in f:
                if 'index' in f: continue
                copyfile(os.path.join(tosort_fld, fld, f),
                            os.path.join(metadata_fld, f))
            elif '.avi' in f:
                os.rename(os.path.join(tosort_fld, fld, f),
                            os.path.join(tosort_fld, fld, fld+'.avi'))
                copyfile(os.path.join(tosort_fld, fld, fld+'.avi'), 
                            os.path.join(video_fld, fld+'.avi'))
            else:
                raise ValueError('Could not proess file with format: ', os.path.split(f)[-1])
    print('... task completed')