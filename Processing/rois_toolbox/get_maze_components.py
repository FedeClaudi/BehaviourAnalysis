import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np

def get_rois_from_templates():
    pass


def get_maze_configuration_transitions(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError('Could not open videofile')

    prev_frame = None
    pixel_difference = []
    print('Processing')
    framen = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        framen += 1
        print(framen)

        if not framen % 10 == 0: continue

        if framen < 10000: 
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_color = np.mean(gray)
        # gray = cv2.threshold(gray, mean_color+100, 255, cv2.THRESH_BINARY)[1]

        if prev_frame is None:
            prev_frame = gray
        else:
            diff = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            pixel_difference.append(np.mean(thresh))
            prev_frame = gray

    print('Processed: {} frames'.format(len(pixel_difference)))
    plt.plot(pixel_difference)
    plt.show()


if __name__ == "__main__":
    test_file = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\to_sort\\181114_CA3155_1\\cam1.avi'
    # os.chd('Users/federicoclaudi/Desktop')
    get_maze_configuration_transitions(test_file)


