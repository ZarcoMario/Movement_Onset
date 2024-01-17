'''
This is a simple resampling method as I no longer have an active Matlab license.
I assume the GUI is using an efficient resampling function
'''
import numpy as np
import pandas as pd
from scipy import interpolate


def resample_splines(t: np.array, x: np.array, y: np.array, z: np.array):
    n_steps = t.size
    tck, _ = interpolate.splprep([x, y, z], u=t, s=0)
    t_resampled = np.linspace(t.min(), t.max(), n_steps)
    x_resampled, y_resampled, z_resampled = interpolate.splev(t_resampled, tck)
    return pd.DataFrame.from_dict({'t': t_resampled, 'x': x_resampled, 'y': y_resampled, 'z': z_resampled})


if __name__ == "__main__":

    '''
    Test 'resample_splines' using real data
    '''

    import os
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # VR-S1 Data
    p_ = 1
    participant_ = r"\P" + str(p_).zfill(2)
    path_ = os.path.dirname(os.getcwd()) + r"\VR-S1" + participant_ + r"\S001"

    path_results = path_ + r"\trial_results.csv"
    results = pd.read_csv(path_results, usecols=['start_time'])
    start_time = results['start_time'].to_numpy()

    trial_number = 128

    path_trial = path_ + r"\trackers" + r"\controllertracker_movement_T" + str(trial_number).zfill(3) + ".csv"

    # Load Raw Data
    raw_data = pd.read_csv(path_trial, usecols=['time', 'pos_x', 'pos_y', 'pos_z'])

    # Adjust to Zero
    t_raw = raw_data['time'].to_numpy() - start_time[trial_number - 1]
    x_raw = raw_data['pos_x'].to_numpy()
    y_raw = raw_data['pos_y'].to_numpy()
    z_raw = raw_data['pos_z'].to_numpy()

    # Resampling
    resampled_data = resample_splines(t_raw, x_raw, y_raw, z_raw)

    t = resampled_data['t'].to_numpy()
    x = resampled_data['x'].to_numpy()
    y = resampled_data['y'].to_numpy()
    z = resampled_data['z'].to_numpy()

    # Sanity check
    gs = GridSpec(4, 1)
    fig = plt.figure(figsize=(16, 8))

    # --- t --- #
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.diff(t_raw), '.-', label='Non-uniformly Sampled')
    ax.plot(np.diff(t), '.-', label='Resampled')
    ax.grid(True)
    ax.legend()

    # --- x --- #
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_raw, x_raw, '.')
    ax.plot(t, x, '.')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.grid(True)

    # --- y --- #
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t_raw, y_raw, '.')
    ax.plot(t, y, '.')
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.grid(True)

    # --- z --- #
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(t_raw, z_raw, '.')
    ax.plot(t, z, '.')
    ax.set_xlabel('t')
    ax.set_ylabel('z')
    ax.grid(True)

    plt.tight_layout()
    plt.show()