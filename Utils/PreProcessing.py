import copy
import scipy.signal as sig
import numpy as np


def butter_filter_lowpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='lowpass')
    return sig.filtfilt(b, a, copy.deepcopy(data))


def butter_filter_highpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='highpass')
    return sig.filtfilt(b, a, copy.deepcopy(data))


