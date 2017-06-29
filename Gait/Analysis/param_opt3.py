from os.path import join
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from hyperopt import space_eval
from hyperopt import fmin, tpe, Trials
from Gait.Pipeline.StepDetection import StepDetection
import Gait.config as c
from Gait.Analysis.config_param_search import space_single_side

##########################################################################################################
with open(join(c.pickle_path, 'metadata_sample'), 'rb') as fp:
    sample = pickle.load(fp)
with open(join(c.pickle_path, 'acc'), 'rb') as fp:
    acc = pickle.load(fp)
ids = sample[sample['StepCount'].notnull()].index.tolist()  # use only samples with step count
sc = StepDetection(acc, sample)
sc.select_specific_samples(ids)


class SingleSide:
    def __init__(self, p_params=None, signal=None, smoothing=None, mva_win=None, butter_freq_single_side=None, peaktype=None,
                 p1_sc=None, p2_sc=None, p1_pu=None, p2_pu=None, remove_weak_signals=None):
        self.sc = None
        self.signal = p_params['signal']
        self.smoothing = p_params['smoothing']
        self.mva_win = p_params['mva_win']
        self.butter_freq_single_side = p_params['butter_freq_single_side']
        self.peaktype = p_params['peaktype']
        self.p1_sc = p_params['p1_sc']
        self.p2_sc = p_params['p2_sc']
        self.p1_pu = p_params['p1_pu']
        self.p2_pu = p_params['p2_pu']
        self.remove_weak_signals = p_params['remove_weak_signals']

    def fit(self, X, y, **kwargs):
        X.select_signal(kwargs['Signal'])
        if kwargs['Smoothing'] == 'mva':
            X.mva(win_size=kwargs['mva_win'])
        elif kwargs['Smoothing'] == 'butter':
            X.bf(p_type='lowpass', order=5, freq=kwargs['butter_freq_single_side'])
        X.mean_normalization()

        if kwargs['peaktype'] == 'scipy':
            X.step_detect_single_side_wpd_method(side='lhs', peak_type=kwargs['peaktype'], p1=kwargs['p1_sc'],
                                                 p2=kwargs['p1_sc'] + kwargs['p2_sc'])
        elif kwargs['peaktype'] == 'peak_utils':
            X.step_detect_single_side_wpd_method(side='lhs', peak_type=kwargs['peaktype'], p1=kwargs['p1_pu'], p2=kwargs['p2_pu'])
        if kwargs['remove_weak_signals']:
            X.remove_weak_signals(kwargs['weak_signal_thresh'])
        self.sc = X


def objective(params):
    pipe = SingleSide(params)
    shuffle = KFold(n_splits=5, shuffle=True)
    x = sc
    y = sc.res['sc_true']
    score = cross_val_score(pipe, x, y, cv=shuffle, scoring='mean_squared_error', n_jobs=1)
    return score.mean()


##########################################################################################################
space = space_single_side

# The Trials object will store details of each iteration
trials = Trials()

# Run the hyperparameter search using the tpe algorithm
best = fmin(objective, space, algo=tpe.suggest, max_evals=5, trials=trials)

##########################################################################################################
# Results

print(best)
# Get the values of the optimal parameters
best_params = space_eval(space, best)


