import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('./')   

from database.dj_config import start_connection
from database.Tables_definitions import *
start_connection()

def run():
    posedata = pd.DataFrame(TrackingData.Body.fetch())
    snoutdata = pd.DataFrame(TrackingData.Snout.fetch())
    taildata = pd.DataFrame(TrackingData.Tail3.fetch())
    for i in range(100):
        first = posedata.iloc[i]
        snt = snoutdata.iloc[i]
        tl = taildata.iloc[i]

        f, ax = plt.subplots()
        ax.set(facecolor=[.2, .2, .2])

        x, y = first['overview'][5000:, 0], first['overview'][5000:, 1]
        plt.plot(x, y)

        x, y = snt['overview'][5000:, 0], snt['overview'][5000:, 1]
        plt.plot(x, y, alpha=.5)

        
        x, y = tl['overview'][5000:, 0], tl['overview'][5000:, 1]
        plt.plot(x, y, alpha=.5)
        
        plt.show()



if __name__ == "__main__":
    run()