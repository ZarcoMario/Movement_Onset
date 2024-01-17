import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from movement_onset_detection import onset_detection
from resample import resample_splines
from derivative import calculate_velocity
from filter import butter_lowpass_filter

# VR-S1 Data

width, height = 16, 8

fig_x = plt.figure(figsize=(width, height))
gs_x = GridSpec(2, 10)
fig_x.suptitle("x")

fig_y = plt.figure(figsize=(width, height))
gs_y = GridSpec(2, 10)
fig_y.suptitle("y")

fig_z = plt.figure(figsize=(width, height))
gs_z = GridSpec(2, 10)
fig_z.suptitle("z")

# range_ = [i for i in range(1, 44 + 1) if i != 17]

# PARTICIPANTS
for i, p_ in enumerate(range(1, 20 + 1)):

    print("Participant", p_)

    participant_ = r"\P" + str(p_).zfill(2)
    path_ = os.path.dirname(os.getcwd()) + r"\VR-S1" + participant_ + r"\S001"

    path_results = path_ + r"\trial_results.csv"
    results = pd.read_csv(path_results, usecols=['start_time', 'initial_time'])
    start_time = results['start_time'].to_numpy()
    initial_time = results['initial_time'].to_numpy()
    # Time threshold for onset_detection method
    t_threshold = initial_time - start_time

    congruency = pd.read_csv(path_results, usecols=['trial_num', 'congruency'], index_col=0)
    accuracy = pd.read_csv(path_results, usecols=['trial_num', 'accuracy'], index_col=0)

    # Congruent Trials
    x_ct, y_ct, z_ct = [], [], []

    # Incongruent Trials
    x_it, y_it, z_it = [], [], []

    ax_x = fig_x.add_subplot(gs_x[i])
    ax_x.grid(True)

    ax_y = fig_y.add_subplot(gs_y[i])
    ax_y.grid(True)

    ax_z = fig_z.add_subplot(gs_z[i])
    ax_z.grid(True)

    # TRIALS
    for trial_number in range(17, 284 + 1, 1):

        # print('Trial', trial_number)

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
        x_res = resampled_data['x'].to_numpy()
        y_res = resampled_data['y'].to_numpy()
        z_res = resampled_data['z'].to_numpy()

        # Filtering.
        # This is a test. A filter might be helpful but this step is not necessary
        cutoff_fq = 10
        x = butter_lowpass_filter(x_res, cutoff_fq, 90, 2)
        y = butter_lowpass_filter(y_res, cutoff_fq, 90, 2)
        z = butter_lowpass_filter(z_res, cutoff_fq, 90, 2)
        # x, y, z = x_res, y_res, z_res

        # Note: step is typically the same for all trials (e.g. if 90 Hz, step=1/90)
        # Although step is similar across trials, step is quickly calculated here
        step = t[1] - t[0]
        vx = calculate_velocity(step, x)
        vy = calculate_velocity(step, y)
        vz = calculate_velocity(step, z)
        '''
        Movement Onset Time Detection
        Parameters (see https://www.frontiersin.org/articles/10.3389/neuro.20.002.2009/full)
            Each segment with m data points (m < N); m = delta_T / Ts, and m * Ts <= Ts
            delta_T is the time interval set according to prior knowledge of human reaction time and feedback loop delay
            Ts is the uniform sapling time interval
        '''
        delta_T = 0.1  # 100 ms.
        Ts = step
        m = int(delta_T / Ts) - 1
        tm = m * Ts

        res = onset_detection(m, x, z, t, vx, vz, t_th=t_threshold[trial_number - 1], vel_th=0.6)
        # Note: Movement onset Time does not necessarily correspond to a sample
        to = res[0]

        # Interpolation to find the corresponding location and velocity
        idx_ub = np.argwhere(t > to).T[0][0]
        idx_lb = np.argwhere(t < to).T[0][-1]

        x_to = np.interp(to, (t[idx_lb], t[idx_ub]), (x[idx_lb], x[idx_ub]))
        y_to = np.interp(to, (t[idx_lb], t[idx_ub]), (y[idx_lb], y[idx_ub]))
        z_to = np.interp(to, (t[idx_lb], t[idx_ub]), (z[idx_lb], z[idx_ub]))

        if accuracy.loc[trial_number, 'accuracy'] == 1:

            if congruency.loc[trial_number, 'congruency']:
                x_ct.extend([x_to])
                y_ct.extend([y_to])
                z_ct.extend([z_to])
            else:
                x_it.extend([x_to])
                y_it.extend([y_to])
                z_it.extend([z_to])

    x_ct, y_ct, z_ct = np.array(x_ct), np.array(y_ct), np.array(z_ct)
    x_it, y_it, z_it = np.array(x_it), np.array(y_it), np.array(z_it)

    a_ = 0.5

    ax_x.hist(x_ct, color='blue', density=True, alpha=a_)
    ax_y.hist(y_ct, color='blue', density=True, alpha=a_)
    ax_z.hist(z_ct, color='blue', density=True, alpha=a_)

    ax_x.hist(x_it, color='orange', density=True, alpha=a_)
    ax_y.hist(y_it, color='orange', density=True, alpha=a_)
    ax_z.hist(z_it, color='orange', density=True, alpha=a_)


fig_x.tight_layout()
fig_y.tight_layout()
fig_z.tight_layout()

fig_x.savefig("vr_s1_x.png")
fig_y.savefig("vr_s1_y.png")
fig_z.savefig("vr_s1.z.png")
# plt.show()