
import sys
sys.path.append('./')

from Utilities.imports import *

from scipy import signal

from Analysis.Behaviour.utils.experiments_analyser import ExperimentsAnalyser

def save_T_tracking_plot(aligned_trials, yth=250, shelter_location = [500, 800],
                    ax_kwargs=dict(ylim=[120, 370], xlim=[425, 575])):
    if sys.platform == "darwin":
        save_fld = "/Users/federicoclaudi/Dropbox (UCL - SWC)/Rotation_vte/plots/Tplatf/Trials"
    else:
        save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\plots\\Tplatf\\Trials"

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
        # same but for shelter
        shelter_angle = angle_between_points_2d_clockwise([x[0], y[0]], shelter_location)
        
        # plot tracking
        frames = np.arange(0, 20, 5)
        tracking_ax.plot([tx, x], [ty, y], color=grey, lw=1, zorder=50)
        tracking_ax.plot([x, nx], [y, ny], color=grey, lw=1, zorder=50)
        tracking_ax.plot([nx, sx], [ny, sy], color=red, lw=1, zorder=50)
        tracking_ax.scatter(sx, sy, color=red, zorder=99, s=15, alpha=.8)

        # Plot line between start and escape locations + shelter
        tracking_ax.plot([x[0], x[-1]], [y[0], y[-1]], color=green, lw=2)
        tracking_ax.plot([x[0], 500 + (500 - x)[-1]], [y[0], y[-1]], color=orange, lw=2)
        tracking_ax.plot([x[0], shelter_location[0]], [y[0], shelter_location[1]], color=blue, lw=2)

        # plot angles
        time, maxt = np.arange(len(trial['body_orientation'])), len(trial['body_orientation'])
        polax.plot(np.radians(trial['body_orientation']), time, lw=4, color=white, zorder=98)

        # Plot line towards the escape
        polax.plot([np.radians(escape_angle), np.radians(escape_angle)], [0, maxt], lw=2, color=green)
        polax.plot([np.radians(escape_angle+180), np.radians(escape_angle+180)], [0, maxt], lw=2, color=green)
        
        polax.plot([np.radians(escape_angle_opposite), np.radians(escape_angle_opposite)], [0, maxt], lw=2, color=orange)
        polax.plot([np.radians(escape_angle_opposite+180), np.radians(escape_angle_opposite+180)], [0, maxt], lw=2, color=orange)
        
        polax.plot([np.radians(shelter_angle), np.radians(shelter_angle)], [0, maxt], lw=2, color=blue)
        polax.plot([np.radians(shelter_angle+180), np.radians(shelter_angle+180)], [0, maxt], lw=2, color=blue)
        
        # Plot a line representing the animals initial orientation
        polax.plot([np.radians(trial['body_orientation'][0]), np.radians(trial['body_orientation'][0])], [0, 10], lw=8, color=magenta, zorder=99)
        
        # Set axes props
        tracking_ax.axhline(yth, color=white, lw=2)
        _ = tracking_ax.set(title="Trcking. Escape on {}".format(trial.escape_side), 
                            facecolor=[.2, .2, .2],  **ax_kwargs) 
        _ = polax.set(ylim=[0, above_yth],facecolor=[.2, .2, .2], title="Angle of body over time, below Yth")
        polax.grid(False)

        save_figure(f, os.path.join(save_fld, "{}_{}.png".format(trial.condition, counter)))
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
    aligned_trials  = get_T_data(load=False, median_filter=True)
    save_T_tracking_plot(aligned_trials)