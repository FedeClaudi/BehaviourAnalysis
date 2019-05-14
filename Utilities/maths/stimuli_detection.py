import sys
sys.path.append('./')

from Utilities.imports import *
from Utilities.Maths.filtering import *

"""
    Collection of functions that analyse AI time series data to find the start and end of stimuli delivered through Mantis
"""

def find_audio_stimuli(data, th, sampling_rate):
    above_th = np.where(data>th)[0]
    peak_starts = [x+1 for x in np.where(np.diff(above_th)>sampling_rate)]
    stim_start_times = above_th[peak_starts]
    try:
        stim_start_times = np.insert(stim_start_times, 0, above_th[0])
    except:
        raise ValueError
    else:
        return stim_start_times



def find_visual_stimuli(data, th, sampling_rate):
    # Filter the data to remove high freq noise, then take the diff and thereshold to find changes
    filtered  = butter_lowpass_filter(data, 75, int(sampling_rate/2))
    d_filt = np.diff(filtered)
    starts = [ x for x in np.where(d_filt < - 0.0005)[0] if x > 1000] # skipping first 1000 dpoints to ignore distortions due to filtering and diff
                                                                        # assuming no stim delivered at the very start of the recording
    ends = [x for x in np.where(d_filt > 0.0003)[0] if x > 1000]

    # make sure you have only 1 start and 1 end per stim
    unique_starts = [starts[0]]
    unique_starts.extend([starts[s] for s in np.where(np.diff(starts)>1)[0]])
    unique_ends = [ends[0]]
    unique_ends.extend([starts[s] for s in np.where(np.diff(ends)>1)[0]])

    assert len(unique_starts) == len(unique_ends), "ops"

    # ? For debugging
    # f, ax = plt.subplots()
    # ax.plot(filtered, color='r')
    # ax.plot(butter_lowpass_filter(np.diff(filtered), 75, int(sampling_rate/2)), color='g')
    # ax.scatter(starts, [0.25 for i in starts], c='r')
    # ax.scatter(ends, [0 for i in ends], c='k')

    # plt.show()
    
    # Return as a list of named tuples
    stim = namedtuple("stim", "start end")
    return [stim(s,e) for s,e in zip(unique_starts, unique_ends)]



