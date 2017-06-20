# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 15:49:58 2017

@author: awagner
"""


import numpy as np

''' Normlize a signal: '''
def normlize_sig(sig):
    y = (sig  - np.min(sig))/(np.max(sig)-np.min(sig))
    return y