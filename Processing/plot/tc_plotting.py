import sys
sys.path.append('./')

from shutil import copyfile

from Utilities.imports import *

from Processing.rois_toolbox.rois_stats import get_roi_at_each_frame, get_arm_given_rois, convert_roi_id_to_tag
from Utilities.video_and_plotting.video_editing import Editor 
# from Modelling.bayesian.hierarchical_bayes_v2 import Modeller as Bayesian


traces_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\HB traces'


def pr_sym_vs_asmy_get_traces():
    bayes = Bayesian()
    traces_fld = 'D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\HB traces'


    # Get data
    asym_exps = ["PathInt2", "PathInt2-L"]
    sym_exps = ["Square Maze", "TwoAndahalf Maze"]

    asym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in asym_exps] for arm in arms]
    sym = [arm for arms in [get_trials_by_exp(e, 'true', ['escape_arm']) for e in sym_exps] for arm in arms]

    asym_origins = [arm for arms in [get_trials_by_exp(e, 'true', ['origin_arm']) for e in asym_exps] for arm in arms]
    sym_origins = [arm for arms in [get_trials_by_exp(e, 'true', ['origin_arm']) for e in sym_exps] for arm in arms]

    asym_sessions = [s for sessions in [get_sessuids_given_experiment(e) for e in asym_exps] for s in sessions]
    sym_sessions = [s for sessions in [get_sessuids_given_experiment(e) for e in sym_exps] for s in sessions]


    """
        LOOK AT EFFECT OF ARM OF ORIGIN
    """
    asym_r_ori = [e for o,e in zip(asym_origins, asym) if 'Right' in o]
    sym_r_ori = [e for o,e in zip(sym_origins, sym) if 'Right' in o]

    asym_l_ori = [e for o,e in zip(asym_origins, asym) if 'Left' in o]
    sym_l_ori = [e for o,e in zip(sym_origins, sym) if 'Left' in o]

    # DO SOME MODELLING
    if 1 == 1:
        asym_r_ori_int = [1 if 'Right' in e else 0 for e in asym_r_ori]
        asym_l_ori_int = [1 if 'Right' in e else 0 for e in asym_l_ori]
        trace, D, dp, t, tp = bayes.model_two_distributions(asym_r_ori_int, asym_l_ori_int)
        bayes.save_trace(trace, os.path.join(traces_fld, 'asym_origin.pkl'))
        

        sym_r_ori_int = [1 if 'Right' in e else 0 for e in sym_r_ori]
        sym_l_ori_int = [1 if 'Right' in e else 0 for e in sym_l_ori]
        trace,  D, dp, t, tp = bayes.model_two_distributions(sym_r_ori_int, sym_l_ori_int)
        bayes.save_trace(trace, os.path.join(traces_fld, 'sym_origin.pkl'))

    """
        LOOK AT THE EFFECT OF X POSITION
    """

    # Plot the probs of escaping left and right based on the position at stim onset
    asym_tracking = [arm for arms in [get_trials_by_exp(e, 'true', ['tracking_data']) for e in asym_exps] for arm in arms]
    sym_tracking = [arm for arms in [get_trials_by_exp(e, 'true', ['tracking_data']) for e in sym_exps] for arm in arms]

    asym_position_onset = [1 if 480 > tr[0, 0, 0] else 2 if 520 < tr[0, 0, 0] else 0 for tr in asym_tracking ]
    sym_position_onset = [1 if 480 > tr[0, 0, 0] else 2 if 520 < tr[0, 0, 0] else 0 for tr in sym_tracking ]

    asym_position_onset_pos = [tr[0, :2, 0] for tr in asym_tracking if (480 > tr[0, 0, 0] or 520 < tr[0, 0, 0])]
    sym_position_onset_pos = [tr[0, :2, 0] for tr in sym_tracking if (480 > tr[0, 0, 0] or 520 < tr[0, 0, 0])]

    asym_l_pos, asym_r_pos = [e for i,e in enumerate(asym) if asym_position_onset[i]==1], [e for i,e in enumerate(asym) if asym_position_onset[i]==2]
    sym_l_pos, sym_r_pos = [e for i,e in enumerate(sym) if sym_position_onset[i]==1], [e for i,e in enumerate(sym) if sym_position_onset[i]==2]

    # do some MODELLING
    if 1 == 1:
        asym_l_pos_int, asym_r_pos_int = [1 if 'Right' in e else 0 for e in asym_l_pos], [1 if 'Right' in e else 0 for e in asym_r_pos]
        sym_l_pos_int, sym_r_pos_int = [1 if 'Right' in e else 0 for e in sym_l_pos], [1 if 'Right' in e else 0 for e in sym_r_pos]
        

        trace,  D, dp, t, tp = bayes.model_two_distributions(asym_r_pos_int, asym_l_pos_int)
        bayes.save_trace(trace, os.path.join(traces_fld, 'asym_position.pkl'))

        trace,  D, dp, t, tp = bayes.model_two_distributions(sym_r_pos_int, sym_l_pos_int)
        bayes.save_trace(trace, os.path.join(traces_fld, 'sym_position.pkl'))


    """
        LOOK AT THE EFFECT OF ORIENTATION
    """

    asym_body_pos, asym_tail_pos = np.vstack([tr[0, :2, 0] for tr in asym_tracking ]), np.vstack([tr[0, :2, -1] for tr in asym_tracking ])
    sym_body_pos, sym_tail_pos = np.vstack([tr[0, :2, 0] for tr in sym_tracking ]), np.vstack([tr[0, :2, -1] for tr in sym_tracking ])

    asym_orient = calc_angle_between_vectors_of_points_2d(asym_body_pos.T, asym_tail_pos.T)
    sym_orient = calc_angle_between_vectors_of_points_2d(sym_body_pos.T, sym_tail_pos.T)

    asym_rorient, asym_lorient = [e for i,e in enumerate(asym) if asym_orient[i] <= 90-22.5], [e for i,e in enumerate(asym) if 180 >= asym_orient[i] >= 90+22.5]
    sym_rorient, sym_lorient = [e for i,e in enumerate(sym) if sym_orient[i] <= 90-22.5], [e for i,e in enumerate(sym) if 180 >= sym_orient[i] >= 90+22.5]
    
    asym_rorient_int, asym_lorient_int= [1 if 'Right' in e else 0 for e in asym_rorient], [1 if 'Right' in e else 0 for e in asym_lorient]
    sym_rorient_int, sym_lorient_int= [1 if 'Right' in e else 0 for e in sym_rorient], [1 if 'Right' in e else 0 for e in sym_lorient]

    trace, D, dp, t, tp = bayes.model_two_distributions(asym_rorient_int, asym_lorient_int)
    bayes.save_trace(trace, os.path.join(traces_fld, 'asym_orientation.pkl'))

    trace, D, dp, t, tp = bayes.model_two_distributions(sym_rorient_int, sym_lorient_int)
    bayes.save_trace(trace, os.path.join(traces_fld, 'sym_orientation.pkl'))

    plt.show()

""""
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
"""



def plot_two_dists_kde(d0, d1, d2, title, l1=None, l2=None, ax=None, no_ax_lim=False):
        colors = get_n_colors(6)

        if ax is None:
            f, ax = plt.subplots()

        c1, c2 = colors[2], colors[3]

        colors = ['k', c1, c2]
        labels = ['all', l1, l2]
        distributions = [d0, d1, d2]
        alphas = [.25, .8, .8]
        shades = [False, True, True]

        for i, (d,c,l,a,s) in enumerate(zip(distributions, colors, labels, alphas, shades)):
            if d is None: continue

            d_mean_ci = mean_confidence_interval(d)
            d_range = percentile_range(d)

            if not no_ax_lim:
                sns.kdeplot(d, ax=ax, shade=s, color=c, linewidth=2, alpha=a, clip=[0, 1], label=l)
            else:
                sns.kdeplot(d, ax=ax, shade=s, color=c, linewidth=2, alpha=a, label=l)

            y = (-i * .5) - .5
            ax.plot([d_range.low, d_range.high], [y, y], color=c, linewidth=4, label='5-95 perc')
            ax.scatter(d_mean_ci.mean, y, s=100, alpha=.35, c=c)
            ax.plot([d_mean_ci.interval_min, d_mean_ci.interval_max], [y, y], color=c, linewidth=8, label='Mean C.I.')
            ax.axhline(0, color='k', linewidth=2)

        if not no_ax_lim:
            ax.set(title=title, xlim=[-0.01, 1.01],  xlabel='p(R)', ylabel='pdf')
        else:
            ax.set(title=title, xlabel='p(R)', ylabel='pdf')  
        ax.legend()

        if ax is None:
            f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\{}.svg".format(title.strip().split('-')[0]), format="svg")

def plotter():
    bayes = Bayesian()
    types = ['asym', 'sym']
    variables = ['origin', 'position', 'orientation']
    names = [t+'_'+v for t in types for v in variables]

    f, axarr = plt.subplots(nrows=2, ncols=len(variables))
    axarr = axarr.flatten()

    for name, ax in zip(sorted(names), axarr):
        trace = bayes.load_trace(savename=os.path.join(traces_fld, name+'.pkl'))
        tot_trace = bayes.load_trace(savename=os.path.join(traces_fld, 'hb_trace.pkl'))
        if 'asym' in name:
            tt = tot_trace['p_asym_grouped'].values
        else:
            tt = tot_trace['p_sym_grouped'].values

        plot_two_dists_kde(tt, trace['p_d1'].values, trace['p_d2'].values, name, 'R', 'L', ax=ax)

    f.savefig("D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\plots\\alternative_hp_stuff_things.svg", format="svg")

def number_of_trials_per_session():
    ntrials = []
    good_exp = ['PathInt', 'PathInt2', 'PathInt2-L', 'PathInt2 - L', 'Square Maze', 'TwoAndahalf Maze']
    for session in set((Sessions).fetch("uid")):
        try:
            exp = (AllTrials & "session_uid={}".format(session)).fetch("experiment_name")[0]
            if not exp in good_exp: continue
            ntrials.append((AllTrials & "session_uid={}".format(session)).fetch("number_of_trials")[0])
        except:
            pass
    
    f, ax = plt.subplots()
    ax.hist(ntrials, color='k', bins=18)
    ax.axvline(np.median(ntrials), color='r', linewidth=3, linestyle='--')
    a = 1

def plot_distributions():
    f, axarr = plt.subplots(ncols=4)

    binom = np.random.binomial(25, .5, size=1000000)
    # binom = stats.binom(1000, .5)
    sns.kdeplot(binom/25, color='k', linewidth=3, ax=axarr[0], clip=[0, 1], bw=.1)

    beta = np.random.beta(2, 9, size=1000000)
    sns.kdeplot(beta, color='k', linewidth=3, ax=axarr[1], clip=[0, 1])

    uniform = np.random.uniform(0, 1, 10000000)
    sns.kdeplot(uniform, color='k', linewidth=3, ax=axarr[2], clip=[0, 1])

    gamma = np.random.gamma(2, 2, size=1000000)
    sns.kdeplot(gamma, color='k', linewidth=3, ax=axarr[3])

    titles = ['binomial', 'beta', 'uniform', 'gamma']
    for t,ax in zip(titles, axarr):
        if t != 'gamma':
            ax.set(title=t, xlim=[-0.05, 1.05])
        else: 
            ax.set(title=t, xlim=[-0.05, 20.05])


def plot_expl_asym_vs_sym():
    asym_exps = ['PathInt2', 'PathInt2 - L']
    asym_exploration_tracking = []
    for e in asym_exps:
        asym_exploration_tracking.extend([x[:, :, 0] for x in (AllExplorations & "experiment_name='{}'".format(e)).fetch("tracking_data")])
    asym_exploration_tracking = np.vstack(asym_exploration_tracking)
    asym_exploration_tracking = asym_exploration_tracking[(asym_exploration_tracking[:, -1] != 0) & (asym_exploration_tracking[:, -1] != 1)]

    sym_exps = ['Square Maze', 'TwoAndahalf Maze']
    sym_exploration_tracking = []
    for e in sym_exps:
        sym_exploration_tracking.extend([x[:, :, 0] for x in (AllExplorations & "experiment_name='{}'".format(e)).fetch("tracking_data")])
    sym_exploration_tracking = np.vstack(sym_exploration_tracking)
    sym_exploration_tracking = sym_exploration_tracking[(sym_exploration_tracking[:, -1] != 0) & (sym_exploration_tracking[:, -1] != 1)] # ? remove times where its on shelter and thereat platf

    f, axarr = plt.subplots(ncols=2)
    axarr[0].hexbin(asym_exploration_tracking[:, 0], asym_exploration_tracking[:, 1], mincnt=1, bins='log')
    axarr[1].hexbin(sym_exploration_tracking[:, 0], sym_exploration_tracking[:, 1], mincnt=1, bins='log')

    

    asym_platforms = asym_exploration_tracking[:, -1]
    asym_on_right = asym_exploration_tracking[(asym_platforms == 18) | (asym_platforms == 6) | (asym_platforms==13)].shape[0]
    asym_on_left = asym_exploration_tracking[(asym_platforms == 17) | (asym_platforms == 3) | (asym_platforms==11) | (asym_platforms==2) | (asym_platforms==8)].shape[0]

    sym_platforms = sym_exploration_tracking[:, -1]
    sym_on_right = sym_exploration_tracking[(sym_platforms == 18) | (sym_platforms == 6) | (sym_platforms==13)].shape[0]
    sym_on_left = sym_exploration_tracking[(sym_platforms == 17) | (sym_platforms == 3) | (sym_platforms==12) ].shape[0]


    f, ax = plt.subplots()

    ax.plot([0, 1], [asym_on_left/asym_on_right, asym_on_right/asym_on_right], color='k')
    ax.plot([0, 1], [asym_on_left/asym_on_right, asym_on_right/asym_on_right], 'o', color='k')

    ax.plot([0, 1], [sym_on_left/sym_on_right, sym_on_right/sym_on_right], color='r')
    ax.plot([0, 1], [sym_on_left/sym_on_right, sym_on_right/sym_on_right], 'o', color='r')

    ax.set(title='Exploration time on left vs right', ylabel='Normed time spent', xticks=[0, 1], xticklabels=['left', 'right'], ylim=[0, 1.8])

    plt.show()

    a =1 



def plot_hierarchical_bayes_posteriors():
    bayes = Bayesian()
    trace = bayes.load_trace("Processing\\modelling\\bayesian\\hierarchical_v2.pkl")

    f, axarr = plt.subplots(ncols=5)

    for c in list(trace.columns):
        if 'asym_prior' in c:
            col, i = 'k', 1
        elif 'sym_prior' in c:
            col, i = 'r', 2
        else: continue
        
        sns.kdeplot(trace[c].values, color=col, linewidth=3, ax=axarr[i], clip=[0, 1])

    plot_two_dists_kde(None, trace.mode_hyper__0.values, trace.mode_hyper__1.values, 'Pop Mode', ax=axarr[0])

  

    axarr[4].hist(np.random.gamma(0.01, 1/0.01, 100000), bins=500, density=True, alpha=.3, label='prior')
    axarr[4].hist(trace.concentration_hyper__0.values, bins=500, density=True, alpha=.3, label='asym')
    axarr[4].hist(trace.concentration_hyper__1.values, bins=500, density=True, alpha=.3, label='sym')
    axarr[4].legend()
    for ax in axarr[1:-2]:
        ax.set(xlim=[0, 1], ylim=[0, 7])



    diff = trace.mode_hyper__0.values - trace.mode_hyper__1.values
    sns.kdeplot(diff, color=col, linewidth=3, ax=axarr[3], clip=[0, 1])
    d_mean_ci = mean_confidence_interval(diff)
    d_range = percentile_range(diff)

    y = (-i * .5) - .5
    axarr[3].plot([d_range.low, d_range.high], [y, y], color='k', linewidth=4, label='5-95 perc')
    ax.scatter(d_mean_ci.mean, y, s=100, alpha=.35, c='r')
    axarr[3].plot([d_mean_ci.interval_min, d_mean_ci.interval_max], [y, y], color='k', linewidth=8, label='Mean C.I.')









def mbv2_behav_plots(plot=False, baseline = True):
    trials_data = "Processing\\trials_analysis\\data\\mbv2_trials.yml"
    data = load_yaml(trials_data)

    sessions = list(data.keys())
    # Get the number of escapes on each arm in baseline trials
    arms = ["B0", "A0", "A1",  "B1"]
    escapes = {a:0 for a in arms}

    for session in sessions:
        if len(data[session][0]) != len(data[session][1]): raise ValueError(session, len(data[session][0]), len(data[session][1]))
        if session not in data.keys(): continue
        arms, maze_states = data[session][:2]
        for arm, maze_state in zip(arms, maze_states):
            if arm.lower() == "n": continue

            if baseline:
                if maze_state != 0: 
                    escapes[arm] += 1  # still count it
                    break
            else:
                if maze_state == 0: continue

            escapes[arm] += 1

    tot_trials = np.sum(list(escapes.values()))
    escapes_probs = {a:k/tot_trials for a,k in escapes.items()}

    if plot:
        f, ax = plt.subplots()
        ax.bar([0, 1, 2, 3], escapes_probs.values())
        ax.set(ylim=[0, 0.5])
        plt.show()

    return escapes_probs


def mbv2_closes_plots():
    trials_data = "Processing\\trials_analysis\\mbv2_trials.yml"
    data = load_yaml(trials_data)
    sessions = list(data.keys())

    arms = ["B0", "A0", "A1",  "B1"]
    escapes = {a: [0, 0, 0, 0, 0, 0] for a in arms}

    f,ax = plt.subplots()
    for i, session in enumerate(sessions): 
        # if "190328" in session: continue
        if not 1 in data[session][1]: continue
        else:
            cleaned = [(a, m) for a,m in zip(*data[session][:2]) if a.lower() != "n"]

            first_closed =  [m for a,m in cleaned].index(1)

            ax.plot(np.add(2*i, [1 if "A" in a else 0 for a,m in cleaned[first_closed:]]))
            ax.axhline(2*i-0.1, color='k')
            ax.axhline(2*i+1.1, color='k')

            for x in np.arange(6):
                try:
                    escapes[cleaned[first_closed+x-2][0]][x] += 1
                except:
                    pass
    plt.show()
    escapes['A'] = np.add(escapes['A0'], escapes['A1'])
    escapes['B'] = np.add(escapes['B0'], escapes['B1'])

    baseline_probs = mbv2_behav_plots(baseline=True)

    baseline_probs['A'] = np.add(baseline_probs['A0'], baseline_probs['A1'])
    baseline_probs['B'] = np.add(baseline_probs['B0'], baseline_probs['B1'])

    n_trials = [a+b for a,b in zip(escapes['A'], escapes['B'])]

    colors = get_n_colors(6)
    f, ax = plt.subplots()
    for c, (arm, escs) in zip(colors, escapes.items()):
        if "1" in arm or "0" in arm: continue
        ax.plot(escs / n_trials, 'o-', color=c,  label=arm)
        ax.plot(0, baseline_probs[arm], 'o', color=c)

    ax.axhline(0.25, color=[.8, .8, .8], linestyle="--", )

    for x in [0, 1, 2]:
        ax.axvline(x, color=[.8, .8, .8], linestyle="--", )

    ax.legend()
    ax.set(ylim=[0, 1.01], xlim=[-1, 5])
    plt.show()

def mbv2_make_videos(play_videos = False):
    editor = Editor()
    trials_data = "Processing\\trials_analysis\\mbv2_trials.yml"
    video_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\to watch"
    save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\videos"
    
    videos = [v for v in os.listdir(video_fld)]
    data = load_yaml(trials_data)
    sessions = list(data.keys())

    for session in sessions: 
        if not 1 in data[session][1]: continue
        else:
            # cleaned = [(a, m) for a,m in zip(*data[session]) if a.lower() != "n"]

            session_vids = [v for v in videos if session in v]
            
            first_closed =  data[session][1].index(1)
            video = [os.path.join(video_fld, v) for v in videos if session in v][0]

            print("\n\n")
            print("Sesssion: {}\ntrial: {}".format(session, first_closed))
            print("\n\n")

            start = 40*21*(first_closed+1)
            elapse = 20*40
            end = start + elapse

            # editor.play_video(video, faster=False, play_from=start, stop_after=elapse)
        
            editor.trim_clip(video, os.path.join(save_fld, session+'_tri-{}_one_after.mp4'.format(first_closed)), frame_mode = True,
                                start_frame=start, stop_frame=end)

def  mbv2_make_cool_vids():
    editor = Editor()
    trials_data = "Processing\\trials_analysis\\mbv2_trials.yml"
    video_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\to watch"
    save_fld = "D:\\Dropbox (UCL - SWC)\\Rotation_vte\\Presentations\\ThesisCommitte\\videos"
    
    videos = [v for v in os.listdir(video_fld)]
    data = load_yaml(trials_data)
    sessions = list(data.keys())

    for session in sessions:
        if len(data[session]) == 2: continue
        
        else:
            video = [os.path.join(video_fld, v) for v in videos if session in v][0]

            for tr_num in data[session][3]:
                start = 40*21*(tr_num)
                elapse = 20*40
                end = start + elapse        
                savename = os.path.join(save_fld, session+'_tri-{}_cool.mp4'.format(tr_num))
                if  session+'_tri-{}_cool.mp4'.format(tr_num) not in os.listdir(save_fld):
                    editor.trim_clip(video, savename, frame_mode = True,
                                        start_frame=start, stop_frame=end)
        
            
def mbv2_delta_probs_before_after():
    baseline = mbv2_behav_plots(plot=False, baseline=True)
    post = mbv2_behav_plots(plot=False, baseline=False)

    delta = {k:post[k]-v for k,v in baseline.items()}

f, ax = plt.subplots()
ax.bar([0, 1, 2, 3], delta.values())
ax.set(ylim=[-0.5, 0.5])
plt.show()

if __name__ == "__main__":
    # mbv2_behav_plots(plot=True, baseline=True)
    # mbv2_closes_plots()
    # mbv2_make_videos()
    # mbv2_make_cool_vids()
    mbv2_delta_probs_before_after()
    # pr_sym_vs_asmy_get_traces()
    plt.show()

