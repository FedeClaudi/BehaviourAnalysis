stimuli_frames = {}
    cameras = namedtuple('cameras', 'overview threat')
    for stim_time in stim_start_times:
        stimuli_frames[str(stim_time)] = cameras(stim_time/samples_per_frame['overview'], stim_time/samples_per_frame['overview']) 