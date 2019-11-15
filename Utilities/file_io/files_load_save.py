import yaml
import shutil
import os
import subprocess
import sys
from multiprocessing import Process
import numpy as np
from functools import partial
import pandas as pd

sys.path.append("./")

def load_yaml(file):
        if not isinstance(file, str): raise ValueError('Invalid input argument')
        with open(file, 'r') as f:
                try:
                    loaded = yaml.full_load(f)
                except: loaded = yaml.load(f)
        return loaded

def save_yaml(path, obj, mode='w'):
    try:
        with open(path, mode) as f:
            yaml.dump(obj, f)
    except: return False
    else: return True

def load_tdms_from_winstore(filetomove):
        print('Moving ', filetomove, ' with size ', np.round(os.path.getsize(filetomove)/1000000000, 2), ' GB')
        temp_dest = "M:\\"
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
        shutil.copy(filetomove, dest)

        print('Moved {} to {}'.format(filetomove, dest))
        return dest


def load_feather(path):
    return pd.read_feather(path)

def save_df(df, filepath):
        df.to_pickle(filepath)

def load_df(filepath):
        return pd.read_pickle(filepath)

if __name__ == "__main__":
    path = "Z:\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata\\as_pandas\\180223_CA503_1.ft"
    loaded = load_feather(path)
    print(loaded)

