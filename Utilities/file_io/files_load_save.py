import yaml
import shutil
import os
import subprocess
import sys
import pyfastcopy
from multiprocessing import Process
import numpy as np
from functools import partial

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                loaded = yaml.load(f)
        return loaded


# def copyWithSubprocess(cmd, source, dest):
        # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        #                       stderr=subprocess.PIPE, shell=True)
        # subprocess.call("mv {} {}".format(source, dest), shell=True)
        # p = subprocess.Popen(['mv', source, dest], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)



def load_tdms_from_winstore(filetomove):
        print('Moving ', filetomove, ' with size ', np.round(os.path.getsize(filetomove)/1000000000, 2), ' GB')
        temp_dest = "D:\\"
        origin, name = os.path.split(filetomove)
        dest = os.path.join(temp_dest, name)
        if name in os.listdir(temp_dest):
                s1 = os.path.getsize(filetomove)
                s2 = os.path.getsize(dest)
                if s1 == s2:
                        print('File was already there')
                        return dest
                else:
                        os.remove(dest)
        # ? python file copying is slow, using subprocess instead
        # ? but suprocess is platform dependant so get right command 


        shutil.copy(filetomove, dest)


        print('Moved {} to {}'.format(filetomove, dest))
        return dest




