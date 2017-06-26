# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:46:45 2017

@author: awagner
"""
from scipy.signal import butter, filtfilt
from future.utils import lmap
import numpy as np
import pywt
import copy


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Input:
    lowcut - low frequency cut
    highcut - high frequency cut
    fs - sample rate
    order - order of filter
    Output:
    b - low freq paramter for filtfilt
    a - high freq paramter for filtfilt
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Input: data - time signal
    lowcut - low frequency cut
    highcut - high frequency cut
    fs - sample rate
    order - order of filter
    Output
    y - the signal after butter filter using lowcut and high cut, fs and order
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data-np.mean(data))
    return y


# Zeev's function - may be redundant with above
def butter_filter_lowpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = butter(order, float(freq)/float(nyq), btype='lowpass')
    return filtfilt(b, a, copy.deepcopy(data))


# Zeev's function - may be redundant with above
def butter_filter_highpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = butter(order, float(freq)/float(nyq), btype='highpass')
    return filtfilt(b, a, copy.deepcopy(data))


def denoise2(data, high_cut):
    """
    Denoise the data with wavelet and butter
    Input:
    data - time signal
    Output:
    result - signal after denosing
    """
    if np.std(data) < 0.01:
        result = data - np.mean(data)
    else:
        result = butter_bandpass_filter(data - np.mean(data), 0.2, high_cut, 50, order=4)
    return result


def denoise(data):
    """
    Denoise the data with wavelet and
    Input:
        data - time signal
    Output:
        result - signal after denoising
    """
    data = data - np.mean(data) + 0.1
    WC = pywt.wavedec(data, 'sym8')
    threshold = 0.01*np.sqrt(2*np.log2(256))
    NWC = lmap(lambda x: pywt.threshold(x, threshold, 'soft'), WC)
    result = pywt.waverec(NWC, 'sym8')
    return result - np.mean(result)


def denoise_signal(signal_data, high_cut=12):
    """
    denoise_Sgnal
    Input:
        signal_data - numpy array
    Output:
        denoised_signal - numpy array of denoised rows
    """
    denoised_signal = lmap(denoise, signal_data)
    return denoised_signal

"""
def fusedlasso(sig, beta, mymatrix):
    Input:
        data - time signal
    Output:
        result - signal after denosing
    sig = np.reshape(sig, 250)
    x = Variable(len(sig))
    #if np.std(sig)<0.05:
    #obj = Minimize(square(norm(x-sig))+tv(mul_elemwise(beta,x)))
    obj = Minimize(square(norm(x-sig))+beta*quad_form(x,mymatrix))
    prob = Problem(obj)
    prob.solve()  # Returns the optimal value.
    res = x.value
    res = np.asarray(res.flatten())
    return res[0]
"""
