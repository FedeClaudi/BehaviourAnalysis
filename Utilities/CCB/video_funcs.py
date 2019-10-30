import cv2
import numpy as np
import scipy.misc
from tqdm import tqdm
import glob
import os


def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j+=1


    background = (background / (j)).astype(np.uint8)
    cv2.imshow('background', background)
    cv2.waitKey(10)
    vid.release()

    return background

# =================================================================================
#              CREATE MODEL ARENA FOR COMMON COORDINATE BEHAVIOUR
# =================================================================================
def model_arena(size):
    ''' NOTE: this is the model arena for the Barnes maze with wall
    this function must be customized for any other arena'''
    # model_arena = cv2.resize(model_arena,size)
    model_arena = cv2.imread('C:\\Users\\Federico\\Desktop\\mazemodel.png')

    points = np.array(([500,500+460-75],[500-460+75,500],[500,500-460+75],[500+460-75,500]))* [size[0]/1000,size[1]/1000]

    # cv2.imshow('model_arena',model_arena)

    return model_arena, points

# =================================================================================
#              IMAGE REGISTRATION GUI
# =================================================================================
def register_arena(background, fisheye_map_location, x_offset, y_offset, arena, arena_points,  savepath, savename):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """

    # create model arena and background
    # arena, arena_points = model_arena(background.shape)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, x_offset, int((map1.shape[0] - background.shape[0]) - x_offset),
                                                y_offset, int((map1.shape[1] - background.shape[1]) - y_offset),cv2.BORDER_CONSTANT, value=0)
        raise NotImplementedError('Border is added just to the right making the correction skewed to one side')
        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        background_copy = background_copy[x_offset:-int((map1.shape[0] - background.shape[0])- x_offset),
                                            y_offset:-int((map1.shape[1] - background.shape[1]) - y_offset)]
    except:
        background_copy = background.copy()
        fisheye_map_location = ''
        # print('fisheye correction not available')

    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False
    use_loaded_points = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1
    # transform_files = glob.glob('*transform.npy')
    transform_files = [os.path.join(savepath, f) for f in os.listdir(savepath) if savename.split('.')[0] in f]
    for file_num, transform_file in enumerate(transform_files[::-1]):

        # USE LOADED TRANSFORM AND SEE IF IT'S GOOD
        loaded_transform = np.load(transform_file, allow_pickle=True)
        M = loaded_transform[0]
        background_data[1] = loaded_transform[1]
        arena_data[1] = loaded_transform[2]

        # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)
        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                       * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]

        # print('Does transform ' + str(file_num+1) + ' / ' + str(len(transform_files)) + ' match this session?')
        # print('\'y\' - yes! \'n\' - no. \'q\' - skip examining loaded transforms. \'p\' - update current transform')
        while True:
            # cv2.imshow('registered background', overlaid_arenas)
            # k = cv2.waitKey(10)
            use_loaded_transform = True
            break
            # if  k == ord('n'):
            #     break
            # elif k == ord('y'):
            #     use_loaded_transform = True
            #     break
            # elif k == ord('q'):
            #     make_new_transform_immediately = True
            #     break
            # elif k == ord('p'):
            #     use_loaded_points = True
            #     break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately or use_loaded_points:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        if not use_loaded_points:
            print('\nSelect reference points on the experimental background image in the indicated order')

            # initialize clicked point arrays
            background_data = [background_copy, np.array(([], [])).T]
            arena_data = [[], np.array(([], [])).T]

            # add 1-2-3-4 markers to model arena
            for i, point in enumerate(arena_points.astype(np.uint32)):
                arena = cv2.circle(arena, (point[0], point[1]), 3, [0, 0, 255], -1)
                arena = cv2.circle(arena, (point[0], point[1]), 4, [0, 255, 255], 1)
                cv2.putText(arena, str(i+1), tuple(point), 0, .55, [0, 0, 255], thickness=2)

                point = np.reshape(point, (1, 2))
                arena_data[1] = np.concatenate((arena_data[1], point))

            # initialize GUI
            cv2.startWindowThread()
            cv2.namedWindow('background')
            cv2.imshow('background', background_copy)
            cv2.namedWindow('model arena')
            cv2.imshow('model arena', arena)

            # create functions to react to clicked points
            cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

            while True: # take in clicked points until four points are clicked
                cv2.imshow('background',background_copy)

                number_clicked_points = background_data[1].shape[0]
                if number_clicked_points == len(arena_data[1]):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # perform projective transform
        # M = cv2.findHomography(background_data[1], arena_data[1])
        M = cv2.estimateAffine2D(background_data[1], arena_data[1], False)[0]
        # M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)
        # M = cv2.getAffineTransform(background_data[1],arena_data[1])

        if not M.any():
            raise ValueError('Could not calculate Rigid Transform')

        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        # registered_background = cv2.warpPerspective(background_copy,M[0],background.shape)

        registered_background = cv2.warpAffine(background_copy, M, background.shape[:2])
        # --------------------------------------------------
        # overlay images
        # --------------------------------------------------
        alpha = .7
        colors = [[150, 0, 150], [0, 255, 0]]
        color_array = make_color_array(colors, background.T.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                 * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print('\nLeft click model arena // Right click model background')
        print('Purple within arena and green along the boundary represent the model arena')
        print('Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step')
        print('Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        print('y: save and use transform')
        print('r: reset transform (left and right click four points to recommence)')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif  k == ord('r'):
                print('Transformation erased')
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
            elif k == ord('q') or k == ord('y'):
                print('Registration completed')
                break

            if update_transform:
                update_transform_data[3] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape)
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data[0] = overlaid_arenas
        save_name = os.path.join(savepath, savename)
        np.save(save_name,[M, update_transform_data[1], update_transform_data[2], fisheye_map_location])

    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location]


# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array



########################################################################################################################
if __name__ == "__main__":
    model_arena(1000)