# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:23:55 2017

@author: awagner
"""

norma = (XXX**2 + YYY**2 + ZZZ**2)**0.5

norma_filt = map(lambda x: butter_bandpass_filter(x - np.mean(x),0.2,3.5,50),norma)
norma_filt12 = map(lambda x: butter_bandpass_filter(x - np.mean(x),0.2,12,50),norma)

Activity_level = np.asarray(map(lambda x: np.mean(np.abs(x)),norma_filt))
Total_acc = np.asarray(map(lambda x: np.mean(np.abs(x)) ,norma_filt12))

tremor_score =  Total_acc - Activity_level

RTS = tremor_score/Activity_level

Activity_level = Activity_level/np.std(Activity_level)
tremor_score = tremor_score/np.std(tremor_score)
RTS = RTS/np.std(RTS)

TagLow = np.column_stack((Activity_level,tremor_score,RTS)) 

