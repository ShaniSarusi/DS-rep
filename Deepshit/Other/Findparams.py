# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:06:15 2017

@author: awagner
"""

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space =  {'learning_rate': hp.uniform('learning_rate',0,0.1),'n_estimators': hp.choice('n_estimators',range(1,50)), 'max_depth': hp.choice('max_depth',range(1,20)), 'min_child_weight': hp.choice('min_child_weight',range(1,5)),
             'gamma': hp.uniform('gamma',0,1),'subsample': hp.uniform('subsample',0.1,0.9), 'C': hp.uniform('C',0,2)}



def optimparam(params, colsample_bytree=1, scale_pos_weight=1,reg_lambda1=0,reg_alpha1=0, 
               clus =[1,2], Men=Men,data_for_pred = TagLow ,symp = Dys,Task_for_model = Task,meta_for_model = meta):
    xg = XGBoost_Classifier2(params['learning_rate'], params['n_estimators'], max_depth = params['max_depth'] , min_child_weight = params['min_child_weight'], 
                             gamma = params['gamma'], subsample = params['subsample'],
            colsample_bytree=1, scale_pos_weight=1,reg_lambda1 = 0,reg_alpha1 = 0)               
    tab= make_pred_withprob(clus =clus, Men=Men,data_for_pred = TagLow ,symp = symp,Task_for_model = Task,meta_for_model = meta,model = xg, C=params['C'])
    #
    tabALL = pd.DataFrame(np.zeros((2,2)))
    pred_total = np.zeros(1)
    for i in range(len(tab)): 
        temp = pd.DataFrame(tab[i])
        colname = temp.columns.values
        rowname = list(temp.index)
        for col in colname:
            for row in rowname:
                tabALL[row][col] = tabALL[row][col] + temp[row][col]       
    #            
    return {'loss': float(tabALL[1][1]+tabALL[0][0])/np.sum(np.sum(tabALL)),
            'status': STATUS_OK,
            'eval_time': time.time(),
            'attachments':{'time_module': pickle.dumps(time.time)}
            }
    
def findandsaveparams(space,max_evals,trials = 0, func = optimparam):
    if  trials == 0:   
            trials = Trials()
    best = fmin(func,
                space=space,
                algo=tpe.suggest,
                max_evals =max_evals ,
                trials=trials)                
    #pickle.dump(trials, open( "/data/iil/home/awagner/PycharmProjects/Deepshit/trailsResult.p", "wb" ) )
    return trials, best
trials, best = findandsaveparams(space,20,trials = 0, func = optimparam)
for t in trials.trials:
    print t['misc']['vals']['x']