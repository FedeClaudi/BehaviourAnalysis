# %%

import os

from tqdm import tqdm
from behaviour.tdms.utils import get_analog_inputs_clean_dataframe



# %%
videos_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\video'
ai_fld = 'Z:\\swc\\branco\\Federico\\raw_behaviour\\maze\\analoginputdata'

sessions = ['200227_CA8755_1', '200227_CA8754_1', '200227_CA8753_1', '200227_CA8752_1',
            '200225_CA8751_1', '200225_CA848_1', '200225_CA8483_1', '200225_CA834_1',
            '200225_CA832_1', '200210_CA8491_1', '200210_CA8482_1', '200210_CA8481_1',
            '200210_CA8472_1', '200210_CA8471_1', '200210_CA8283_1']

videos = [os.path.join(videos_fld, s+'Overview.mp4') for s in sessions]
ais = [os.path.join(ai_fld, s+'.tdms') for s in sessions]

# %%

for ai in tqdm(ais):
    get_analog_inputs_clean_dataframe(ai)

# %%