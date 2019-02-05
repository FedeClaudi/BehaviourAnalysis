import sys
sys.path.append('./')  
from Utilities.file_io.files_load_save import load_yaml
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import namedtuple
import yaml

def get_rois_from_templates(session_name, videopath, templates_fld):
    """ Uses template matching to identify the different components of the maze and their location """
    paths = load_yaml('paths.yml')
    matched_fld = paths['templates_matched']

    # Finds the templates for the session being processed
    def get_templates(templates_fld):
        print('Looking for templates in >>> ', templates_fld)
        # Get the templates
        platf_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'p' in f]
        bridge_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'f' in f]
        maze_templates = [os.path.join(templates_fld, f) for f in os.listdir(templates_fld) if 'bg' in f]
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
            col = [.2, .3, .5]
            templ = cv2.imread(template)
            if len(templ.shape) == 3: templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            w, h = templ.shape[::-1]

            res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)  # ! <-- template matching here
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
        bg = cv2.imread(maze_templates[0])
        if bg.shape[-1] > 2: bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('Could not find maze templates', templates_fld)

    # Calculate the position of the templates and save resulting image
    display, platforms = loop_over_templates(platf_templates, bg)
    display, bridges = loop_over_templates(bridge_templates, display, bridge_mode=True)
    cv2.imwrite(os.path.join(matched_fld, '{}.png'.format(session_name[1])), display)

    # Return locations of the templates
    dic = {**platforms, **bridges}
    return dic

def display_maze_components():
    background = cv2.imread('Utilities/video_and_plotting/mazemodel.png')
    background = cv2.resize(background, (1000, 1000))

    components = load_yaml('Processing/rois_toolbox/template_components.yml')

    for pltf, (center, radius) in components['platforms'].items():
        cv2.circle(background, tuple(center), radius, (255, 0, 0), 3)

    for bridge, pts in components['bridges'].items():
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(background, [pts], True, (0, 0, 255), 2)

    cv2.imshow('bg', background)
    cv2.waitKey(0)


def user_click_rois_locations():
    def register_click(event,x,y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
                # clicks = np.reshape(np.array([x, y]),(1,2))

                data[0].append(x)
                data[1].append(y)
                

    clicks_data = [[], []]

    maze_model = cv2.imread('Utilities\\video_and_plotting\\mazemodel.png')
    maze_model = cv2.resize(maze_model, (1000, 1000))
    maze_model = cv2.cvtColor(maze_model,cv2.COLOR_RGB2GRAY)

    paths = load_yaml('paths.yml')
    rois = load_yaml(paths['maze_model_templates']) 
    rois_names = list(rois.keys())

    cv2.startWindowThread()
    cv2.namedWindow('background')
    cv2.imshow('background', maze_model)

    cv2.setMouseCallback('background', register_click, clicks_data)  # Mouse callback

    while True:
        number_clicked_points = len(clicks_data[0])
        k = cv2.waitKey(10)
        if number_clicked_points < len(rois_names):
            print('Please define position of roi: {}'.format(rois_names[number_clicked_points]))
            if k == ord('u'):
                print('Updating')
                # Update positions
                for x,y in zip(clicks_data[0], clicks_data[1]):
                    cv2.circle(maze_model, (x, y), 10, 0, -1)
                    cv2.imshow('background', maze_model)

            elif k == ord('q'):
                break
        else:
            break
    


    save_name = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Maze_templates\\UserSelectedMazeModelTemplates.yml'

    rois = {n:(x,y) for n, x, y in zip(rois_names, clicks_data[0], clicks_data[1])}
    with open(save_name, 'w') as outfile:
        yaml.dump(rois, outfile, default_flow_style=False)

    f, ax = plt.subplots()
    ax.imshow(maze_model)
    for name, x, y in zip(rois_names, clicks_data[0], clicks_data[1]):
        ax.plot(x, y, 'o', label=name)
    ax.legend()
    plt.show()



if __name__ == "__main__":
    # display_maze_components()
    user_click_rois_locations()


