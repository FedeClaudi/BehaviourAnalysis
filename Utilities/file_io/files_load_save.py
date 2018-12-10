import yaml
import shutil
import os

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                loaded = yaml.load(f)
        return loaded


def load_tdms_from_winstore(filetomove):
        temp_dest = "M:\\"
        origin, name = os.path.split(filetomove)
        dest = os.path.join(temp_dest, name)
        if name in os.listdir(temp_dest):
                print('File was already there')
        else:
                # shutil.copy(filetomove, dest)
                os.system('xcopy"{}""{}"'.format(filetomove, dest))
                print('Moved {} to {}'.format(filetomove, dest))
        return dest
