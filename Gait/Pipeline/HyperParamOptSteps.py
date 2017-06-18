from Gait.Pipeline.StepDetection import StepDetection
from os.path import join
import pickle
import Gait.config as config
import numpy as np
import copy
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import OrderedDict


class HyperParamOptSteps:
    def __init__(self, data_acc, data_sample, p_params, ids):
        self.params = p_params
        self.sc = StepDetection(data_acc, data_sample)
        self.sc.select_specific_samples(ids)
        self.rand = None
        self.res = None

    def choose_random_params(self):
        self.rand = {}
        for key in self.params.keys():
            self.rand[key] = random.choice(self.params[key])

    def random_search(self, n=10, p_path=None):
        self.choose_random_params()
        self.res = pd.DataFrame(index=range(n), columns=list(self.rand.keys()) + ['rmse'])
        x = copy.copy(self.sc)
        for i in range(n):
            print("\rRunning random search optimization " + str(i + 1) + " from " + str(n))
            self.choose_random_params()
            self.res.iloc[i] = list(self.rand.values()) + ['NoResYet']

            r = self.rand
            s = None
            s = copy.deepcopy(x)
            s.select_signal(r['Signal'])
            if r['Smoothing'] == 'mva':
                s.mva(win_size=r['mva_win'])
            elif r['Smoothing'] == 'butter':
                s.bf(p_type='lowpass', order=5, freq=r['butter_freq_single_side'])
            s.mean_normalization()

            if r['peaktype'] == 'scipy':
                s.step_detect_single_side_wpd_method(side='lhs', peak_type=r['peaktype'], p1=r['p1_sc'], p2=r['p1_sc'] + r['p2_sc'])
            elif r['peaktype'] == 'peak_utils':
                s.step_detect_single_side_wpd_method(side='lhs', peak_type=r['peaktype'], p1=r['p1_pu'], p2=r['p2_pu'])
            if r['remove_weak_signals']:
                s.remove_weak_signals(r['weak_signal_thresh'])
            for j in range(s.res.shape[0]):
                s.res.set_value(s.res.index[j], 'sc3_lhs', len(s.res.iloc[j]['idx3_lhs']))

            rmse = sqrt(mean_squared_error(s.res['sc_true'], s.res['sc3_lhs']))
            self.res.iloc[i]['rmse'] = rmse
        if p_path is not None:
            self.res.to_csv(p_path)

if __name__ == '__main__':
    single_side_params = {
        'Signal': ['norm'],
        #'Smoothing': ['mva'],
        'Smoothing': ['mva', 'butter'],
        'mva_win': np.arange(5, 50, 5),
        'butter_freq_single_side': np.arange(1, 6, 0.5),
        'peaktype': ['scipy', 'peak_utils'],
        #'peaktype': ['scipy'],
        'p1_sc': np.arange(1, 20).tolist(),
        'p2_sc': np.arange(1, 40).tolist(),
        'p1_pu': np.arange(0.1, 1, 0.1),
        'p2_pu': np.arange(5, 50, 5),
        'remove_weak_signals': [False]
        # 'remove_weak_signals': [True, False],
        # 'weak_signal_thresh': np.arange(-1, 0.25, 0.5)
      }

    with open(join(config.pickle_path, 'metadata_sample'), 'rb') as fp:
        sample = pickle.load(fp)
    with open(join(config.pickle_path, 'acc'), 'rb') as fp:
        acc = pickle.load(fp)
    id_nums = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
    hyp = HyperParamOptSteps(acc, sample, single_side_params, id_nums)
    save_path = join(config.results_path, 'single_side_hyp_opt.csv')
    hyp.random_search(n=5, p_path=save_path)

