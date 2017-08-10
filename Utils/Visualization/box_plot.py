# -*- coding: utf-8 -*-
"""
This file contains function(s) for creating box plots.

Created on Thu Aug 10 09:05:23 2017

@author: imazeh
"""

import numpy as np
import matplotlib.pyplot as plt

def create_box_plot(df, x_discrete_variable, y_cont_variable, plt_title=None):
    x_discrete_values = sorted(df[x_discrete_variable].unique().tolist())
    boxes_vals = [np.asarray(df[y_cont_variable][df[x_discrete_variable] == x]) for x in x_discrete_values]
    plt.boxplot(boxes_vals)
    plt.xticks(range(1, len(x_discrete_values)+1), [str(int(x)) for x in x_discrete_values])
    plt.xlabel(x_discrete_variable)
    plt.ylabel(y_cont_variable)
    if plt_title:
        plt.title(plt_title)
    plt.show()
    return