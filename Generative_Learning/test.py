import matplotlib
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-3.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)

Z = 10.0 * (Z2 - Z1)

line_colours = ('BlueViolet', 'Crimson', 'ForestGreen',
        'Indigo', 'Tomato', 'Maroon')

line_widths = (1, 1.5, 2, 2.5, 3, 3.5)

plt.figure()
CS = plt.contour(X, Y, Z, levels=[0],                        # add 6 contour lines
                 linewidths=line_widths,            # line widths
                 colors = line_colours)