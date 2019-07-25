import sys
sys.path.append('./')   # <- necessary to import packages from other directories within the project
from Utilities.imports import *
from Processing.plot.plot_distributions import plot_fitted_curve, dist_plot
import statsmodels.api as sm

from scipy.signal import find_peaks, resample

class rtAnalysis:
    def __init__(self):
        pass

    def inspect_rt_metric(self, load=False, plot=True):
        def get_first_peak(x, **kwargs):
            peaks, _ = find_peaks(x, **kwargs)
            if not np.any(peaks): return 0
            return peaks[0]

        def get_above_th(x, th):
            peak = 0
            while peak <= 0 and th>0:
                try:
                    peak = np.where(x > th)[0][0]
                except:
                    peak =  0
                th -= .1
            return peak

        if not load:
            # ? th def
            bodyth, snouth, rtth = 6, 6, 2.5

            data = self.merge_conditions_trials(list(self.conditions.values()))

            if plot:
                f = plt.subplots(sharex=True)

                grid = (2, 5)
                mainax = plt.subplot2grid(grid, (0, 0), colspan=2, rowspan=2)
                bodyax = plt.subplot2grid(grid, (0, 2), colspan=2)
                snoutax = plt.subplot2grid(grid, (1, 2), colspan=2)
                scatterax = plt.subplot2grid(grid, (0, 4))
                histax = plt.subplot2grid(grid, (1, 4))

                mainax.set(title="$speed_{snout} - speed_{body}$", ylim=[-6, 12], xlabel="time (frames)", ylabel="ratio", xlim=[0, 80])
                bodyax.set(ylim=[-2.1, 15], title="body",  xlabel="time (frames)", ylabel="speed (a.u.)", xlim=[0, 80])
                snoutax.set(ylim=[-2.1, 15], title="snout", xlabel="time (frames)", ylabel="speed (a.u.)", xlim=[0, 80])
                scatterax.set(xlabel="body peaks", ylabel="snout peaks", xlim=[0, 120], ylim=[0, 120])
                histax.set(title="reaction times", xlabel="time (s)", ylabel="density")

            datadict = {"trialid":[], "rt_frame":[], "fps":[], "rt_s":[], "rt_frame_originalfps":[]}
            bodypeaks, snoutpeaks, rtpeaks = [], [], []
            for n, (i, trial) in tqdm(enumerate(data.iterrows())):
                # Get data
                start, end = trial.stim_frame, int(np.ceil(trial.stim_frame + (trial.time_out_of_t * trial.fps)))
                
                if end-start < 5: continue

                allsess_body = (TrackingData.BodyPartData & "bpname='body'" & "recording_uid='{}'".format(trial.recording_uid)).fetch("tracking_data")[0]
                allsess_snout = (TrackingData.BodyPartData & "bpname='snout'" & "recording_uid='{}'".format(trial.recording_uid)).fetch("tracking_data")[0]
                body, snout = allsess_body[start:end, :].copy(), allsess_snout[start:end, :].copy()

                # ? remove tracking errors
                body[:, 2][np.where(body[:, 2] > 25)] = np.nan
                snout[:, 2][np.where(snout[:, 2] > 25)] = np.nan

                # If filmed at 30fps, upsample
                if trial.fps < 40:
                    new_n_frames = np.int(body.shape[0] / 30 * 40)
                    body = resample(body, new_n_frames)
                    snout = resample(snout, new_n_frames) 

                    body[:, 2], snout[:, 2] = body[:, 2]/30*40, snout[:, 2]/30*40

                # Get speeds
                bs, ss = line_smoother(body[:, 2],  window_size=11, order=3,), line_smoother(snout[:, 2],  window_size=11, order=3,)
                rtmetric = ss-bs

                # Get first peak
                bpeak, speak, rtpeak = get_above_th(bs, bodyth), get_above_th(ss, snouth), get_above_th(rtmetric, rtth)

                # Append to dictionary
                if bpeak > 0  and speak > 0:
                    rt = rtpeak
                elif bpeak == 0 and speak > 0:
                    rt = speak
                elif bpeak > 0 and speak == 0:
                    rt = bpeak
                else:
                    rt = None

                #  data = {"trialid":[], "rt_frame":[], "fps":[], "rt_s":[], "rt_frame_originalfps":[]}
                if rt is not None:
                    datadict["trialid"].append(trial.trial_id)
                    datadict["rt_frame"].append(rt)
                    datadict["fps"].append(trial.fps)
                    datadict["rt_s"].append(rt/40)
                    datadict["rt_frame_originalfps"].append(int(np.ceil(rt/40 * trial.fps)))
                else:
                    datadict["trialid"].append(trial.trial_id)
                    datadict["rt_frame"].append(np.nan)
                    datadict["fps"].append(trial.fps)
                    datadict["rt_s"].append(np.nan)
                    datadict["rt_frame_originalfps"].append(np.nan)
                
                # Append to listdir
                bodypeaks.append(bpeak)
                snoutpeaks.append(speak)
                rtpeaks.append(rtpeak)

                # Plot
                if plot:
                    bodyax.plot(bs, color=green, alpha=.2)
                    bodyax.scatter(bpeak, bs[bpeak], color=green, alpha=1)

                    snoutax.plot(ss, color=red, alpha=.2)
                    snoutax.scatter(speak, ss[speak], color=red, alpha=1)

                    mainax.plot(rtmetric, color=magenta, alpha=.2)
                    mainax.scatter(rtpeak, rtmetric[rtpeak], color=magenta, alpha=1)

                    scatterax.scatter(bpeak, speak, color=white, s=100, alpha=.4)

            if plot:
                scatterax.plot([0, 200], [0, 200], **grey_line)
                mainax.axhline(rtth, **grey_line)
                snoutax.axhline(snouth, **grey_line)
                bodyax.axhline(bodyth, **grey_line)

                # Plot KDE of RTs
                kde = sm.nonparametric.KDEUnivariate(datadict["rt_s"])
                kde.fit(bw=.1) # Estimate the densities
                histax.fill_between(kde.support, 0, kde.density, alpha=.2, color=lightblue, lw=3,zorder=10)
                histax.plot(kde.support, kde.density, alpha=1, color=lightblue, lw=3,zorder=10)

                # Plot KDE of the peaks 
                for i, (ax, peaks, color) in enumerate(zip([mainax, snoutax, bodyax], [rtpeaks, snoutpeaks, bodypeaks], [magenta, red, green])):
                    kde = sm.nonparametric.KDEUnivariate(np.array(peaks).astype(np.float))
                    kde.fit(bw=1) # Estimate the densities
                    if i == 0:
                        x, y, z = kde.support, kde.density/np.max(kde.density)*2 - 6, -6
                    else:
                        x, y, z = kde.support, kde.density/np.max(kde.density)*1.5 - 2, -2
                    ax.fill_between(x, z, y, alpha=.2, color=color, lw=3,zorder=10)
                    ax.plot(x, y, alpha=1, color=color, lw=3,zorder=10)

            # Save to pickle
            datadf = pd.DataFrame.from_dict(datadict)
            datadf.to_pickle(os.path.join(self.metadata_folder, "reaction_time.pkl"))
        else:
            datadf = pd.read_pickle(os.path.join(self.metadata_folder, "reaction_time.pkl"))
        return datadf