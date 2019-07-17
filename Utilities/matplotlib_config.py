import matplotlib as mpl
import sys

if sys.platform == "darwin":
    mpl.use("Qt5Agg")

# Set up matplotlib
mpl.rcParams['text.color'] = "white"

mpl.rcParams['figure.figsize'] = [20, 16]
mpl.rcParams['figure.facecolor'] = [.2, .2, .2]
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.facecolor'] = [.1, .1, .1]

mpl.rcParams['figure.subplot.left'] = .1
mpl.rcParams['figure.subplot.right'] = .9
mpl.rcParams['figure.subplot.bottom'] = .2
mpl.rcParams['figure.subplot.top'] = .9
mpl.rcParams['figure.subplot.wspace'] = .1
mpl.rcParams['figure.subplot.hspace'] = .1

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.numpoints'] = 2
mpl.rcParams['legend.fontsize'] = 'medium'
mpl.rcParams['legend.framealpha'] = .8
mpl.rcParams['legend.scatterpoints'] = 3
mpl.rcParams['legend.edgecolor'] = 'red'
# mpl.rcParams['legend.facecolor'] = [.2, .2, .2]

mpl.rcParams['axes.facecolor'] = [.2, .2, .2]
mpl.rcParams['axes.edgecolor'] = "white"
mpl.rcParams['axes.labelcolor'] = "white"
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"

mpl.rcParams['image.aspect'] = "auto"

font = {'family' : 'Courier New',
        'weight' : 'bold',
        'size'   : 22}

mpl.rc('font', **font)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-pastel')