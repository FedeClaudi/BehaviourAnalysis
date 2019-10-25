
import sys
sys.path.append('./')

from Utilities.imports import *

from scipy import signal

from Analysis.Behaviour.experiments_analyser import ExperimentsAnalyser


def get_angles(x, y, sx, sy, nx, ny, tx, ty):
    body_lower = calc_angle_between_vectors_of_points_2d(np.vstack([tx, ty]), np.vstack([x, y])) # tail -> body
    body_upper = calc_angle_between_vectors_of_points_2d(np.vstack([x, y]), np.vstack([nx, ny])) # body -> neck
    body_long = calc_angle_between_vectors_of_points_2d(np.vstack([tx, ty]), np.vstack([nx, ny])) # tail -> neck
    head_angles = calc_angle_between_vectors_of_points_2d(np.vstack([nx, ny]), np.vstack([sx, sy])) # neck -> snout
    body_angles = np.nanmedian(np.vstack([body_lower, body_upper, body_long]), 0)

    return head_angles, body_angles

def get_T_data(load=False, median_filter=False):
    if sys.platform == "darwin":
        savename = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/analysis_metadata/threatplatform/aligned_T_tracking.pkl"
    else:
        savename = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\analysis_metadata\\threatplatform\\aligned_T_tracking.pkl"

    if load:
        return pd.read_pickle(savename)

    # Getting data
    ea = ExperimentsAnalyser(load=False,  naive=None, lights=1, escapes=True, escapes_dur=True,  shelter=True, 
                        agent_params={'kernel_size':3, 'model_path':'PathInt2_old.png', 'grid_size':1000})

    # Get Threat Platform data
    ea.prep_tplatf_trials_data(filt=False, remove_errors=True, speed_th=None)

    xth, yth, thetath = 10, 50, 5
    upfps = 1000

    aligned_trials = {'trial_id':[], "tracking":[], "tracking_centered":[], "tracking_turn":[], "turning_frame":[], "direction_of_movement":[],
                        "escape_side":[], "body_orientation":[], "body_angvel":[], "head_orientation":[],
                        "s_tracking":[], "n_tracking":[], "t_tracking":[]}

    for i, (condition, trials) in enumerate(ea.trials.items()):
        if condition == "maze1" or condition == "maze4": continue  # Only work on trials with the catwalk

        # loop over trials
        for n, (ii, trial) in enumerate(trials.iterrows()):
            # Get body, snout tail tracking
            rawx, rawy = trial.threat_tracking[:, 0], trial.threat_tracking[:, 1]
            x, y = trial.threat_tracking[:, 0]-trial.threat_tracking[0, 0], trial.threat_tracking[:, 1]-trial.threat_tracking[0, 1]
            s = trial.threat_tracking[:, 2]

            tx, ty = trial.tail_threat_tracking[:, 0], trial.tail_threat_tracking[:, 1]
            nx, ny = trial.neck_threat_tracking[:, 0], trial.neck_threat_tracking[:, 1]
            sx, sy = trial.snout_threat_tracking[:, 0], trial.snout_threat_tracking[:, 1]

            # If escaping on the left, flipt the x tracking
            if "left" in trial.escape_arm.lower():  
                x -= x
                sx -= sx
                tx -= tx
                left = True
            else:
                left = False


            if median_filter:
                x, y = signal.medfilt(x, kernel_size=5), signal.medfilt(y, kernel_size=5)
                sx, sy = signal.medfilt(sx, kernel_size=5), signal.medfilt(sy, kernel_size=5)
                nx, ny = signal.medfilt(nx, kernel_size=5), signal.medfilt(ny, kernel_size=5)
                tx, ty = signal.medfilt(tx, kernel_size=5), signal.medfilt(ty, kernel_size=5)

            # get time
            t = np.arange(len(x))
            upt = upsample_signal(40, upfps, t)

            # Get direction of movement
            dir_of_mvt = calc_angle_between_points_of_vector(np.vstack([x, y]).T)

            # get point of max distance from straightlinr
            dist = [calc_distance_between_point_and_line([[x[0], y[0]], [x[-1], y[-1]]], [_x, _y]) 
                            for _x,_y in zip(x, y)]
            mdist = np.argmax(dist)

            # Get body and head orientation
            # TODO FIX these damn angles
            head_angles, body_angles = get_angles(rawx, rawy, sx, sy, nx, ny, tx, ty)
                    

            # Fill in data dict
            aligned_trials['trial_id'].append(trial['trial_id'])
            aligned_trials['tracking'].append(np.vstack([rawx, rawy, s]).T)
            aligned_trials['s_tracking'].append(np.vstack([sx, sy]).T)
            aligned_trials['n_tracking'].append(np.vstack([nx, ny]).T)
            aligned_trials['t_tracking'].append(np.vstack([tx, ty]).T)
            aligned_trials['tracking_centered'].append(np.vstack([x, y, s]).T)
            aligned_trials['tracking_turn'].append(np.vstack([x-x[mdist], y-y[mdist], s]).T)
            aligned_trials['turning_frame'].append(mdist)
            aligned_trials['direction_of_movement'].append(dir_of_mvt)
            aligned_trials['body_orientation'].append(body_angles)
            aligned_trials['head_orientation'].append(head_angles)
            aligned_trials['body_angvel'].append(calc_ang_velocity(body_angles))

            if left:  
                aligned_trials['escape_side'].append("left")
            else:
                aligned_trials['escape_side'].append("right")

    aligned_trials = pd.DataFrame.from_dict(aligned_trials)

    # save df to file
    aligned_trials.to_pickle(savename)
    return aligned_trials

def save_T_tracking_plot(aligned_trials):
    if sys.platform == "darwin":
        save_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/plots/Tplatf/Trials"
    else:
        save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\Tplatf\\Trials"

    
    # thresholds
    yth = 250

    for counter, (i, trial) in tqdm(enumerate(aligned_trials.iterrows())):
        # setup figure
        f = plt.figure(figsize=(16, 16), facecolor=white)
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1]) 

        tracking_ax = plt.subplot(gs[0, 0])
        polax = plt.subplot(gs[0, 1], projection='polar')

        # Get right aligned tracking data
        x, y, s = trial.tracking[:, 0], trial.tracking[:, 1], trial.tracking[:, 2]
        sx, nx, tx  = trial.s_tracking[:, 0], trial.n_tracking[:, 0], trial.t_tracking[:,0]
        sy, ny, ty  = trial.s_tracking[:, 1], trial.n_tracking[:, 1], trial.t_tracking[:,1]

        if trial.escape_side == "left":
            x, sx, tx, ns = 500 + (500 - x), 500 + (500 - sx), 500 + (500 - tx), 500 + (500 - nx)

        # Get when mouse goes over the y threshold
        try:
            below_yth = np.where(y <= yth)[0][-1]
        except: 
            below_yth = 0
            above_yth = 0
        else:
            above_yth = below_yth + 1

        # Get angle between mouse initial position and where it leaves T
        escape_angle = angle_between_points_2d_clockwise([x[0], y[0]], [x[-1], y[-1]])
        # same but for other escape arm
        escape_angle_opposite = angle_between_points_2d_clockwise([x[0], y[0]], [500 + (500 - x)[-1], y[-1]])
        
        # plot tracking
        frames = np.arange(0, 20, 5)
        tracking_ax.plot([tx, x], [ty, y], color=grey, lw=1, zorder=50)
        tracking_ax.plot([x, nx], [y, ny], color=grey, lw=1, zorder=50)
        tracking_ax.plot([nx, sx], [ny, sy], color=red, lw=1, zorder=50)
        tracking_ax.scatter(sx, sy, color=red, zorder=99, s=15, alpha=.8)

        # Plot line between start and end location
        tracking_ax.plot([x[0], x[-1]], [y[0], y[-1]], color=green, lw=2)
        tracking_ax.plot([x[0], 500 + (500 - x)[-1]], [y[0], y[-1]], color=orange, lw=2)

        # plot angles
        time, maxt = np.arange(len(trial['body_orientation'])), len(trial['body_orientation'])
        polax.plot(np.radians(trial['body_orientation']), time, lw=4, color=white, zorder=98)
        # polax.scatter(np.radians(trial['head_orientation']), time, s=15, color=red, zorder=99)

        # Plot line towards the escape
        polax.plot([np.radians(escape_angle), np.radians(escape_angle)], [0, maxt], lw=2, color=green)
        polax.plot([np.radians(escape_angle+180), np.radians(escape_angle+180)], [0, maxt], lw=2, color=green)
        polax.plot([np.radians(escape_angle_opposite), np.radians(escape_angle_opposite)], [0, maxt], lw=2, color=orange)
        polax.plot([np.radians(escape_angle_opposite+180), np.radians(escape_angle_opposite+180)], [0, maxt], lw=2, color=orange)

        # Plot a line representing the animals initial orientation
        polax.plot([np.radians(trial['body_orientation'][0]), np.radians(trial['body_orientation'][0])], [0, 10], lw=8, color=magenta, zorder=99)
        
        # Set axes props
        tracking_ax.axhline(yth, color=white, lw=2)
        _ = tracking_ax.set(title="Trcking. Escape on {}".format(trial.escape_side), 
                            facecolor=[.2, .2, .2],  ylim=[120, 370], xlim=[425, 575]) 
        _ = polax.set(ylim=[0, above_yth],facecolor=[.2, .2, .2], title="Angle of body over time, below Yth")
        polax.grid(False)

        save_figure(f, os.path.join(save_fld, "{}.png".format(counter)))
        close_figure(f)

def get_above_yth(y, yth):
    try:
        below_yth = np.where(y <= yth)[0][-1]
    except: 
        below_yth = 0
        above_yth = 0
    else:
        above_yth = below_yth + 1

    return below_yth, above_yth


if __name__ == "__main__":
    aligned_trials  = get_T_data(load=True, median_filter=True)
    save_T_tracking_plot(aligned_trials)