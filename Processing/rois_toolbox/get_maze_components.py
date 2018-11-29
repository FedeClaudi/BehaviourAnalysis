import sys
sys.path.append('./')  
from Utilities.file_io.files_load_save import load_yaml
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np

def get_rois_from_templates(session_name, videopath, templates_fld):
    """ Uses template matching to identify the different components of the maze and their location """
   paths = load_yaml('../paths.yaml')
   matched_fld = paths['templates_matched']

    # Finds the templates for the session being processed
    def get_templates(templates_fld):
        # Get the templates
        platf_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'platform' in f]
        bridge_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'bridge' in f]
        bridge_templates = [b for b in bridge_templates if 'closed' not in b]
        maze_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'maze_config' in f]
        return platf_templates, bridge_templates, maze_templates

    # This function actually does the template matching and returns the location of the best match
    def loop_over_templates(templates, img, bridge_mode=False):
        """
            templates: list of filepaths to templates images
            img: background image
        """
        """ in bridge mode we use the info about the pre-supposed location of the bridge to increase accuracy """
        rois = {}
        point = namedtuple('point', 'topleft bottomright')

        # Set up open CV
        font = cv2.FONT_HERSHEY_SIMPLEX
        if len(img.shape) == 2:  colored_bg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else: colored_bg = img

        # Loop over the templates
        for n, template in enumerate(templates):
            id = os.path.split(template)[1].split('_')[0]
            col = self.colors[id.lower()]
            templ = cv2.imread(template)
            if len(templ.shape) == 3: templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            w, h = templ.shape[::-1]

            res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)  # ! <-- template matching here
            rheight, rwidth = res.shape
            if not bridge_mode:  # platforms
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # location of best template match
                top_left = max_loc
            else:  # take only the relevant quadrant of the frame based on bridge name
                if id == 'Left':
                    res = res[:, 0:int(rwidth / 2)]
                    hor_sum = 0
                elif id == 'Right':
                    res = res[:, int(rwidth / 2):]
                    hor_sum = int(rwidth / 2)
                else:
                    hor_sum = 0

                origin = os.path.split(template)[1].split('_')[1][0]
                if origin == 'T':
                    res = res[int(rheight / 2):, :]
                    ver_sum = int(rheight / 2)
                else:
                    res = res[:int(rheight / 2):, :]
                    ver_sum = 0

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # location of best template match
                top_left = (max_loc[0] + hor_sum, max_loc[1] + ver_sum)
            if max_val < 0.60:
                print('discarded template: {}'.format(template))
                continue

            # Get location and mark on the frame the position of the template
            bottom_right = (top_left[0] + w, top_left[1] + h)
            midpoint = point(top_left, bottom_right)
            rois[os.path.split(template)[1].split('.')[0]] = midpoint
            cv2.rectangle(colored_bg, top_left, bottom_right, col, 2)
            cv2.putText(colored_bg, os.path.split(template)[1].split('.')[0] + '  {}'.format(round(max_val, 2)),
                        (top_left[0] + 10, top_left[1] + 25),
                        font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return colored_bg, rois

    # get templates
    platf_templates, bridge_templates, maze_templates = get_templates(templates_fld)

    # Get background
    if maze_templates:
        img = [t for t in maze_templates if 'default' in t]
        bg = cv2.imread(img[0])
        if bg.shape[-1] > 2: bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Could not find maze templates')

    # Calculate the position of the templates and save resulting image
    display, platforms = loop_over_templates(platf_templates, bg)
    display, bridges = loop_over_templates(bridge_templates, display, bridge_mode=True)
    cv2.imwrite(os.path.join(matched_fld, 'Matched\\{}.png'.format(session_name)), display)

    # Return locations of the templates
    dic = {**platforms, **bridges}
    return dic


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
    test_file = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\raw_data\\to_sort\\181114_CA3155_1\\cam1.mp4'
    # os.chd('Users/federicoclaudi/Desktop')
    get_maze_configuration_transitions(test_file)


