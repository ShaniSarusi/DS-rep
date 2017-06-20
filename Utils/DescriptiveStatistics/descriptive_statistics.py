import numpy as np
from Utils.DataHandling.data_processing import pd_to_np


def mean_and_std(values, round_by=2):
    values = pd_to_np(values)
    # return str(round(np.mean(values), round_by)) + u" \u00B1 " + str(round(np.std(values), round_by))
    return str(round(np.mean(values), round_by)) + ' +- ' + str(round(np.std(values), round_by))


def cv(values, round_by=3):
    values = pd_to_np(values)
    coefficient_of_var = np.std(values) / np.mean(values)
    return str(round(coefficient_of_var, round_by))