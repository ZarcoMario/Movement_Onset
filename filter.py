'''
Simple filter to improve movement onset detection ?
'''
from scipy.signal import butter, filtfilt


def _butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype="low")
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = _butter_lowpass(cutoff, fs, order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
