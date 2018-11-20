import pandas as  pd
import os

filename = 'CollectedData_Federico.h5'
main_path = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\DLC_nets\\Nets\\Maze-Federico-2018-11-16\\labeled-data'
sub_fld = '180604_CA2753_visual-57045'

df = pd.read_hdf(os.path.join(main_path, sub_fld, filename))
print(df)
path_parts = df.index[0].split("\\")
del path_parts[1]
newpath = os.path.join(*path_parts)
# print(df['Federico']['body'].iloc[0].rename(newpath))

for i in df['Federico']['body'].index:
    path_parts = i.split("\\")
    del path_parts[1]
    newpath = os.path.join(*path_parts)
    df.loc[i].rename(newpath, axis='index')
    print(df.loc[i])
# print(df)
# df.to_hdf(os.path.join(main_path, sub_fld, filename),'df_with_missing',format='table', mode='w')


"""
 path_parts = imagename1.split("\\")
            del path_parts[1]
            path_parts[0] = path_parts[0] +'\\'
            imagename1 = os.path.join(*path_parts)

"""