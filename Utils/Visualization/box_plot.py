# -*- coding: utf-8 -*-
"""
This file contains function(s) for creating box plots.

Created on Thu Aug 10 09:05:23 2017

@author: imazeh
"""

import numpy as np
import matplotlib.pyplot as plt


def create_box_plot(df, x_discrete_variable, y_cont_variable, all_possible_x_vals=None, plt_title=None):
    """
    Plot a box-plot of a continuous variable for each value of a discrete variable.
    
    Input:
        df (Pandas DataFrame): Should contain both variables as columns.
        x_discrete_variable (str): The column's name in df.
        y_cont_variable (str): The column's name in df.
        all_possible_x_vals (list): A sorted list of discrete values for the x-axis in the plot.
                                    If None (default), will plot only values which are present in df.
                                    If all_possible_x_vals is provided, all values in this list will be plotted on the
                                    x-axis (even if there will be no box to plot).
        plt_title (str): A title for the plot.
    
    Output:
        The plot is plotted, not returned.
    """
    if all_possible_x_vals:
        x_discrete_values = all_possible_x_vals
    else:
        x_discrete_values = sorted(df[x_discrete_variable].unique().tolist())
    boxes_vals = [np.asarray(df[y_cont_variable][df[x_discrete_variable] == x]) for x in x_discrete_values]
    plt.boxplot(boxes_vals)
    plt.xticks(range(1, len(x_discrete_values)+1), [str(int(x)) for x in x_discrete_values])
    plt.xlabel(x_discrete_variable)
    plt.ylabel(y_cont_variable)
    if plt_title:
        plt.title(plt_title)
    plt.show()


# Example
# if __name__ == '__main__':
#     df = placeholder
#     x_discrete_variable = placeholder
#     y_cont_variable = placeholder
#     create_box_plot(df, x_discrete_variable, y_cont_variable)
