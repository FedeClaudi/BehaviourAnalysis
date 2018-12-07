import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('./')   
from nptdms import TdmsFile
# from database.dj_config import start_connection
# from database.Tables_definitions import *
# start_connection()

def run(f):
        # Get info about the metadata
        metadata = TdmsFile(f)

        print(metadata.groups())
        print(metadata.as_dataframe)

        # Get values to return
        metadata_object = metadata.object()
        props = {n:v for n,v in metadata_object.properties.items()} # fps, width, ...  code below is to print props out

        for name, value in metadata_object.properties.items():
                print("{0}: {1}".format(name, value))
        return

def ForRuben(videofilepath):
        """[loads a video and lets the user select a frame to show]

                Arguments:
                        videofilepath {[str]} -- [path to video to be opened]
        """        
        def get_selected_frame(cap, show_frame):
                cap.set(1, show_frame)
                ret, frame = cap.read() # read the first frame
                return frame

        import cv2   # import opencv
        
        cap = cv2.VideoCapture(videofilepath)
        if not cap.isOpened():
                raise FileNotFoundError('Couldnt load the file')
        
        print(""" Instructions
                        - d: advance to next frame
                        - a: go back to previous frame
                        - s: select frmae
        """)

        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialise showing the first frame
        show_frame = 0
        frame = get_selected_frame(cap, show_frame)

        while True:
                cv2.imshow('frame', frame)

                k = cv2.waitKey(10)

                if k == ord('d'):
                        # Display next frame
                        if show_frame < number_of_frames:
                                show_frame += 1
                elif k == ord('a'):
                        # Display the previous frame
                        if show_frame > 1:
                                show_frame -= 1
                elif k ==ord('s'):
                        selected_frame = int(input('Enter frame number: '))
                        if selected_frame > number_of_frames or selected_frame < 0:
                                print(selected_frame, ' is an invalid option')
                        show_frame = int(selected_frame)

                try:
                        frame = get_selected_frame(cap, show_frame)
                except:
                        raise ValueError('Could not display frame ', show_frame)

if __name__ == "__main__":
        f = 'Z:\\branco\\Federico\\raw_behaviour\\maze\\metadata\\180625_CA2814_2.tdms'
        v = 'Z:\\branco\Federico\\raw_behaviour\\maze\\video\\181115_CA339_2.avi'
        # run(f)
        ForRuben(v)


