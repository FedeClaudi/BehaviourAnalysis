import yaml
import shutil
import os
import subprocess
import sys

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                loaded = yaml.load(f)
        return loaded


def copyWithSubprocess(cmd):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)


def load_tdms_from_winstore(filetomove):
        temp_dest = "M:\\"
        origin, name = os.path.split(filetomove)
        dest = os.path.join(temp_dest, name)
        if name in os.listdir(temp_dest):
                print('File was already there')
        else:
                # ? python file copying is slow, using subprocess instead
                # ? but suprocess is platform dependant so get right command 
                # ! shutil.copy(filetomove, dest)

                if sys.platform.startswith("win"):
                        cmd = ['xcopy', filetomove, dest, '/K/O/X']
                else:
                        cmd = ['cp', filetomove, dest]

                copyWithSubprocess(cmd) # * moving here

                print('Moved {} to {}'.format(filetomove, dest))
        return dest




