'''
Calculate velocity using a Smooth Noise-Robust Differentiator
Three points have been added before and after the actual trajectory
to be able to reasonably calculate the derivative
'''

import numpy as np


def calculate_velocity(step, x):
    x = np.append(x[0] * np.ones(3), np.append(x, x[-1] * np.ones(3)))
    v = (-x[:-6] - 4 * x[1:-5] - 5 * x[2:-4] + 5 * x[4:-2] + 4 * x[5:-1] + x[6:]) / (32 * step)
    return v