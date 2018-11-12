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