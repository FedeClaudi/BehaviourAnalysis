import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def get_rois_from_templates():
    pass


def get_maze_configuration_transitions(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError('Could not open videofile')

    prev_frame = None
    pixel_difference = []
    print('Processing')
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = gray
            pixel_difference.append(0)
        else:
            diff = cv2.absdiff(prev_frame, gray)
            pixel_difference.append(np.sum(diff))
            prev_frame = gray
    print('Processed: {} frames'.format(len(pixel_difference)))
    plt.plot(pixel_difference)
    plt.show()


if __name__ == "__main__":
    test_file = 'Users/federicoclaudi/Desktop/test.mp4'
    # os.chd('Users/federicoclaudi/Desktop')
    get_maze_configuration_transitions(test_file)



