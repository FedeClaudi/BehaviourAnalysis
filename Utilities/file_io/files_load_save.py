import yaml
import shutil
import os
import subprocess
import sys
import pyfastcopy

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                loaded = yaml.load(f)
        return loaded


def copyWithSubprocess(cmd, source, dest):
        # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
         #                       stderr=subprocess.PIPE, shell=True)
        # subprocess.call("mv {} {}".format(source, dest), shell=True)
        # p = subprocess.Popen(['mv', source, dest], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        pass



def load_tdms_from_winstore(filetomove):
        print('Moving ', filetomove, ' with size ', os.path.getsize(filetomove))
        temp_dest = "M:\\"
        origin, name = os.path.split(filetomove)
        dest = os.path.join(temp_dest, name)
        if name in os.listdir(temp_dest):
                print('File was already there')
        else:
                # ? python file copying is slow, using subprocess instead
                # ? but suprocess is platform dependant so get right command 
                shutil.copy(filetomove, dest)

                if sys.platform.startswith("win"):
                        cmd = ['mv', filetomove, temp_dest]
                else:
                        cmd = ['cp', filetomove, temp_dest]
                # copyWithSubprocess(cmd, filetomove, temp_dest) # * moving here

                print('Moved {} to {}'.format(filetomove, dest))
        return dest




