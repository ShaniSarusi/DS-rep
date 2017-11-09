# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:31:41 2017

@author: awagner
"""


features_dict = {'abs_energy': None,
                 'absolute_sum_of_changes': None,
                 'approximate_entropy': [{'m': 2, 'r': 0.25},
                                         {'m': 2, 'r': 0.5},
                                         {'m': 2, 'r': 0.75}],
                 'ar_coefficient': [{'coeff': 0, 'k': 10},
                                    {'coeff': 1, 'k': 10},
                                    {'coeff': 2, 'k': 10},
                                    {'coeff': 3, 'k': 10},
                                    {'coeff': 4, 'k': 10}],
                 'augmented_dickey_fuller': None,
                 'autocorrelation': [{'lag': 0},
                                     {'lag': 1},
                                     {'lag': 2},
                                     {'lag': 4},
                                     {'lag': 8}],
                 'binned_entropy': [{'max_bins': 10}],
                 'kurtosis': None,  
                 'mean_abs_change': None,
                 'mean_change': None,
                 'mean_second_derivate_central': None,
                 'median': None,      
                 'quantile': [{'q': 0.25},
                              {'q': 0.75}],
                 'sample_entropy': None,
                 'skewness': None,
                 'standard_deviation': None,
                 'variance': None
                 }


