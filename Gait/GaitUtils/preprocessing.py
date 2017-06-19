import scipy.signal as sig
import copy


def truncate(data, front, back):
    out = []
    for i in range(len(data)):
        n = len(data[i]['rhs'])
        f = int(n * front/100.0)
        b = int(n * (1.0-back/100.0))
        tmp = {}
        tmp['rhs'] = data[i]['rhs'].iloc[range(f, b)]
        tmp['lhs'] = data[i]['lhs'].iloc[range(f, b)]
        out.append(copy.deepcopy(tmp))
    return out


def butter_filter_bandpass(data, order, sampling_rate, low, high):
    nyq = 0.5 * sampling_rate
    gravity_freq = float(low) / float(nyq)
    tremor_freq = float(high) / float(nyq)
    b, a = sig.butter(order, [gravity_freq, tremor_freq], btype=type)
    return apply_filter(copy.deepcopy(data), b, a)


def butter_filter_highpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='highpass')
    return apply_filter(copy.deepcopy(data), b, a)


def butter_filter_lowpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='lowpass')
    return apply_filter(copy.deepcopy(data), b, a)


def apply_filter(pdata, b, a):
    out = []
    sides = ['lhs', 'rhs']
    axes = ['x', 'y', 'z', 'n']
    for i in range(len(pdata)):
        tmp = pdata[i]
        for side in sides:
            for axis in axes:
                # tmp[side][axis] = sig.lfilter(b, a, pdata[i][side][axis])
                tmp[side][axis] = sig.filtfilt(b, a, pdata[i][side][axis])
        out.append(tmp)
    return out



