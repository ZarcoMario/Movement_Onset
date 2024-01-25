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
fig_x.suptitle("xz")

fig_y = plt.figure(figsize=(width, height))
gs_y = GridSpec(2, 10)
fig_y.suptitle("xy")

fig_z = plt.figure(figsize=(width, height))
gs_z = GridSpec(2, 10)
fig_z.suptitle("yz")

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
    fin_pos_x = pd.read_csv(path_results, usecols=['trial_num', 'fin_pos_x'], index_col=0)
    # print(fin_pos_x)

    # Congruent Trials
    x_ctl, y_ctl, z_ctl = [], [], []
    x_ctr, y_ctr, z_ctr = [], [], []

    # Incongruent Trials
    x_itl, y_itl, z_itl = [], [], []
    x_itr, y_itr, z_itr = [], [], []

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

                # LEFT
                if fin_pos_x.loc[trial_number, 'fin_pos_x'] < 0:
                    x_ctl.extend([x_to])
                    y_ctl.extend([y_to])
                    z_ctl.extend([z_to])
                # RIGHT
                else:
                    x_ctr.extend([x_to])
                    y_ctr.extend([y_to])
                    z_ctr.extend([z_to])
            else:

                # LEFT
                if fin_pos_x.loc[trial_number, 'fin_pos_x'] < 0:
                    x_itl.extend([x_to])
                    y_itl.extend([y_to])
                    z_itl.extend([z_to])
                # RIGHT
                else:
                    x_itr.extend([x_to])
                    y_itr.extend([y_to])
                    z_itr.extend([z_to])

    x_ctl, y_ctl, z_ctl = np.array(x_ctl), np.array(y_ctl), np.array(z_ctl)
    x_itl, y_itl, z_itl = np.array(x_itl), np.array(y_itl), np.array(z_itl)

    x_ctr, y_ctr, z_ctr = np.array(x_ctr), np.array(y_ctr), np.array(z_ctr)
    x_itr, y_itr, z_itr = np.array(x_itr), np.array(y_itr), np.array(z_itr)

    a_ = 0.5

    # ax_x.hist(x_ctl, color='blue', density=True, alpha=a_)
    # ax_y.hist(y_ctl, color='blue', density=True, alpha=a_)
    # ax_z.hist(z_ctl, color='blue', density=True, alpha=a_)
    #
    # ax_x.hist(x_itl, color='orange', density=True, alpha=a_)
    # ax_y.hist(y_itl, color='orange', density=True, alpha=a_)
    # ax_z.hist(z_itl, color='orange', density=True, alpha=a_)
    #
    # ax_x.hist(x_ctr, color='green', density=True, alpha=a_)
    # ax_y.hist(y_ctr, color='green', density=True, alpha=a_)
    # ax_z.hist(z_ctr, color='green', density=True, alpha=a_)
    #
    # ax_x.hist(x_itr, color='magenta', density=True, alpha=a_)
    # ax_y.hist(y_itr, color='magenta', density=True, alpha=a_)
    # ax_z.hist(z_itr, color='magenta', density=True, alpha=a_)

    ax_x.plot(x_ctl, z_ctl, '.', color='blue', alpha=a_, label='CL')
    ax_x.plot(x_itl, z_itl, '.', color='orange', alpha=a_, label='IL')
    ax_x.plot(x_ctr, z_ctr, '.', color='green', alpha=a_, label='CR')
    ax_x.plot(x_itr, z_itr, '.', color='magenta', alpha=a_, label='IR')
    ax_x.legend()

    ax_y.plot(x_ctl, y_ctl, '.', color='blue', alpha=a_, label='CL')
    ax_y.plot(x_itl, y_itl, '.', color='orange', alpha=a_, label='IL')
    ax_y.plot(x_ctr, y_ctr, '.', color='green', alpha=a_, label='CR')
    ax_y.plot(x_itr, y_itr, '.', color='magenta', alpha=a_, label='IR')
    ax_y.legend()

    ax_z.plot(y_ctl, z_ctl, '.', color='blue', alpha=a_, label='CL')
    ax_z.plot(y_itl, z_itl, '.', color='orange', alpha=a_, label='IL')
    ax_z.plot(y_ctr, z_ctr, '.', color='green', alpha=a_, label='CR')
    ax_z.plot(y_itr, z_itr, '.', color='magenta', alpha=a_, label='IR')
    ax_z.legend()

fig_x.tight_layout()
fig_y.tight_layout()
fig_z.tight_layout()

fig_x.savefig("vr_s1_xz.png")
fig_y.savefig("vr_s1_xy.png")
fig_z.savefig("vr_s1.yz.png")
plt.show()