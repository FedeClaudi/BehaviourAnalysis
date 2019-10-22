# ! DEFINE A BUNCH OF COLORS
red      = [.8, .2, .2]
blue     = [.3, .3, .9]
green    = [.2, .8, .2]
orange   = [1,  .6, .0]
pink     = [.7, .4, .5]
magenta  = [1., 0., 1.]
purple   = [.5, 0., .5]
white    = [1., 1., 1.]
black    = [0., 0., 0.]
grey     = [.7, .7, .7]
dark_grey = [.2, .2, .2]
teal     = [0., .7, .7]
lilla    = [.8, .4, .9]
lightblue = [.6, .6, .9]

large_square_fig = (16, 16)

# kwargs for plotting
grey_line = dict(color=grey, lw=4, alpha=.85)


dotted_line = dict(lw=4, ls="--", alpha=.5)
grey_dotted_line = dict(color=grey, lw=4, ls="--", alpha=.5)


big_dot = dict(alpha=.9, s=250)
big_red_dot = dict(c=red, alpha=.9, s=250)
big_blue_dot = dict(c=blue, alpha=.9, s=250)

white_errorbar = dict(fmt='o', markeredgecolor=white, markerfacecolor=white, markersize=10, ecolor=grey, elinewidth=3, capthick=2, alpha=1, zorder=0)


text_axaligned = dict(verticalalignment='bottom', horizontalalignment='right') # transform=ax.transAxes