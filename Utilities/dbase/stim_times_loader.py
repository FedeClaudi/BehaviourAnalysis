import sys
sys.path.append('./')

from Utilities.imports import *

from nptdms import TdmsFile
import os

def load_stimuli_from_tdms(tdmspath, software='behaviour'):
        """ Takes the path to a tdms file and attempts to load the stimuli metadata from it, returns a dictionary
         with the stim times """
        # TODO load metadata
        # Try to load a .tdms
        print('\n Loading stimuli time from .tdms: {}'.format(os.path.split(tdmspath)[-1]))
        try:
            tdms = TdmsFile(tdmspath)
        except:
            raise ValueError('Could not load .tdms file: ', tdmspath)

        if software == 'behaviour':
            stimuli = dict(audio=[], visual=[], digital=[])
            for group in tdms.groups():
                for obj in tdms.group_channels(group):
                    if 'stimulis' in str(obj).lower():
                        for idx in obj.as_dataframe().loc[0].index:
                            if '  ' in idx:
                                framen = int(idx.split('  ')[1].split('-')[0])
                            else:
                                framen = int(idx.split(' ')[2].split('-')[0])

                            if 'visual' in str(obj).lower():
                                stimuli['visual'].append(framen)
                            elif 'audio' in str(obj).lower():
                                stimuli['audio'].append(framen)
                            elif 'digital' in str(obj).lower():
                                stimuli['digital'].append(framen)
                            else:
                                print('                  ... couldnt load stim correctly')
        else:
            raise ValueError('Feature not implemented yet: load stim metdata from Mantis .tdms')
        return stimuli


def load_visual_stim_log(path):
    if not os.path.isfile(path): raise FileExistsError("Couldnt find log file: ", path)
    
    try: 
        log = load_yaml(path)
    except:
        raise ValueError("Could not load: ", path)

    # Transform the loaded data into a dict that can be used for creating a df
    temp_d = {k:[] for k in log[list(log.keys())[0]]}

    for stim_i in sorted(log.keys()):
        for k in temp_d.keys():
            try:
                val = float(log[stim_i][k])
            except:
                val = log[stim_i][k]
            temp_d[k].append(val)
    return pd.DataFrame.from_dict(temp_d).sort_values("stim_count")
