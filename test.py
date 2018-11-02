import pandas as pd
import os
import matplotlib.pyplot as plt


fld = 'D:\\Dropbox (UCL - SWC)\\Dropbox (UCL - SWC)\\Rotation_vte\\processed'
f = '180529_CA2743.h5'
df = pd.read_hdf(os.path.join(fld, f))

plt.figure()
plt.plot(df.body.x.values[10000:20000], df.body.y.values[10000:20000])


a = 1



















