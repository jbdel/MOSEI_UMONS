# import matplotlib.pyplot as plt
# import numpy as np
#
# def plot(d):
#     # An "interface" to matplotlib.axes.Axes.hist() method
#     n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
#                                 alpha=0.7, rwidth=0.85)
#     plt.grid(axis='y', alpha=0.75)
#     plt.title('My Very Own Histogram')
#     maxfreq = n.max()
#     # Set a clean upper y-axis limit.
#     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
#     plt.show()