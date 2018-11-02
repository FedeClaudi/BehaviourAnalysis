import os
import pyexcel
try: import yaml
except: pass

from database.Tables_definitions import *

dj.config['Database.host'] = '127.0.0.1'
dj.config['Database.user'] = 'root'
dj.config['Database.password'] = 'simple'


dj.conn()



# # Populate mouse table
mice = Mouse()

mice_records = 'C:\\Users\\Federico\\Downloads\\FC_animals_records.xlsx'

loaded_excel = pyexcel.get_records(file_name=mice_records)

""" id strain dob sex signle  """
for m in loaded_excel:
    if not m['']: continue
    inputdata = (m[''], m['Strain'],  m['DOB'].strip(), 'M', 'Y')
    try:
        mice.insert1(inputdata)
    except:
        a = 1



# Populate sessions tables
# datalog = 'J:\\Rotation_vte\\analysis\\datalog.xls'
#
#
# raw_data = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data'
# meta, video = 'metadata', 'video'
# metaf = os.path.join(raw_data, meta)
# videof = os.path.join(raw_data, video)
#
# loaded_excel = pyexcel.get_records(file_name=datalog)
#
# sessions = Session()
# sss = sessions.fetch(as_dict=True)
# mice = Mouse()
# mms = mice.fetch(as_dict=True)
# recordings = BehaviourRecording()
# a = 1
#
#
# for session in loaded_excel:
#     mouse_id = session['MouseID']
#     for mouse in mms:
#         idd = mouse['mouse_id']
#         original_idd = mouse['mouse_id']
#         idd = idd.replace('_', '')
#         idd = idd.replace('.', '')
#         if idd.lower() == mouse_id.lower():
#             break
#
#     try:
#         session_data = dict(
#             mouse_id=original_idd,
#             session_uid = str(session['Sess.ID']),
#             session_date = session['Date'],
#             experiment = session['Experiment'],
#             software='B',
#             experimenter='Federico'
#         )
#         # sessions.insert1(session_data)
#
#         subfolders_name = '{}_{}'.format(session['Date'], mouse_id)
#         tdmss = [t for t in os.listdir(metaf) if subfolders_name in t]
#         videos = [v for v in os.listdir(videof) if subfolders_name in v]
#
#         if tdmss:
#             if not len(tdmss) == len(videos):
#                 a = 1
#             else:
#                 for t,v in zip(tdmss, videos):
#                     name = t.split('.')[0]
#                     if len(name.split('_'))<=2:
#                         numb = '1'
#                     else:
#                         numb = name.split('_')[-1]
#
#                     recording_data = dict(
#                         session_uid = session_data['session_uid'],
#                         recording_number = int(numb),
#                         metadata_path = os.path.join(metaf, t),
#                         video_path = os.path.join(videof, v),
#                         tracked='N',
#                         dlc_data='nan'
#                     )
#
#                     try:
#                         recordings.insert1(recording_data)
#                     except:
#                          a =1
#     except:
#          a = 1
# a =1



"""  metadata_path : varchar(128)       # name of the experimenter
     video_path : varchar(128)       # name of the experimenter
     metadata : composite # aa
     bodyparts : composite # aa
     dlc_info : composite # aa
     data : composite # aa"""


#
# @schema
# class Session(dj.Manual):
#      definition = """
#      # experiment session
#      -> Mouse
#      session_uid: varchar(128)    # unique session identifier
#      ---
#      session_date: varchar(128)            # session date
#      software: enum('M', 'B')         # mantis or old behaviour software
#      experimenter: varchar(128)    # name of the experimenter
#      """
#
#
# @schema
# class BehaviourRecording(dj.Manual):
#      definition = """
#      # Individual recordings within a session
#      -> Session
#      recording_number: int               # recording number within a session
#      ---
#      metatadata_path: varchar(128)       # name of the experimenter
#      video_path: varchar(128)       # name of the experimenter
#      """
#
#




