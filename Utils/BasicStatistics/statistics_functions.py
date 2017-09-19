"""
This module contains various statistics functions
"""

import numpy as np
from Utils.DataHandling.data_processing import pd_to_np


def mean_and_std(values, round_by=2):
    """
    Take the values and return the string of mean and standard deviation, rounded to the round_by decimal

    Input:
        values (list, Pandas Series, or np.array): Input for which to calculate the mean and standard deviation
        round_by (int): Decimal to which to round the output mean and standard deviation. Default is 2

    Output:
        out1 (str): Return the string of the mean +- the standard deviation of the input, rounded to the round_by decimal
    """
    values = pd_to_np(values)
    # return str(round(np.mean(values), round_by)) + u" \u00B1 " + str(round(np.std(values), round_by))
    return str(round(np.mean(values), round_by)) + ' +- ' + str(round(np.std(values), round_by))


def cv(values, round_by=3):
    """
    Take the values and return the coefficient of variation (CV), rounded to the round_by decimal

    Input:
        values (list, Pandas Series, or np.array): Input for which to calculate the CV
        round_by (int): Decimal to which to round the output CV. Default is 3

    Output:
        out1 (str): Return the string of the CV value, rounded to the round_by decimal
    """
    values = pd_to_np(values)
    if len(values) == 0:
        return ""
    if np.mean(values) == 0:
        return ""
    coefficient_of_var = np.std(values) / np.mean(values)
    return str(round(coefficient_of_var, round_by))


def mean_absolute_percentage_error(y_true, y_pred, handle_zeros=False, return_std=False):
    """
    Take input and return the mean absolute percentage error

    Input:
        y_true (list or numpy array): list of true values
        y_pred (list or numpy array): list of predicted values
        handle_zeros (boolean): If true (default False), convert all zeros in y_pred to 0.0001 to enable division

    Output:
        mean_val (float): Returns the mean absolute percentage error of the input
        std_val (float): Returns the standard deviation of the absolute percentage error of the input

    """
    y_true = pd_to_np(y_true)
    y_pred = pd_to_np(y_pred)
    if handle_zeros:
        y_true = [0.0001 if y_true[i] == 0 else y_true[i] for i in range(len(y_true))]
    mean_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    std_val = np.std(np.abs((y_true - y_pred) / y_true)) * 100.0
    if return_std:
        return mean_val, std_val
    else:
        return mean_val
