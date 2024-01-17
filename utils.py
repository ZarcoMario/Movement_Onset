import numpy as np


def inside_starting_point(p_i, c_sp, r_sp):
    if np.linalg.norm(p_i - c_sp) < r_sp:
        return True
    else:
        return False