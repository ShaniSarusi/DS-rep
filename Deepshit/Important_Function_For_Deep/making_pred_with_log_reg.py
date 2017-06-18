# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:33:38 2017

@author: awagner
"""


"""
This Function takes as an input clusters of activities symptoms model and data 
and as an output it returns probability for the symptom in inteval

##clus - activity cluster
##Men_number - pateint number
##data_for_pred - data for prediction
##symp - trem, Dys, brady
##Task_for_model - array with the tasks where we build our model
##model - xgboost, svm and more...
"""
def make_pred_withprob( Men_number, data_for_pred, symp, meta_for_model, model):
    print(Men_number)
 
    Tag_train = data_for_pred[np.asarray(meta_for_model.SubjectId) !=  Men_number]; data_train = symp[np.asarray(meta_for_model.SubjectId) !=  Men_number]
    Tag_test = data_for_pred[np.asarray(meta_for_model.SubjectId) ==  Men_number]; data_test = symp[np.asarray(meta_for_model.SubjectId) ==  Men_number]
    ##Fit the model and predict!    
    model.fit(Tag_train, data_train)
    pred = model.predict_proba(Tag_test)
    ####           
    return pred


"""
This Function extract features from probabilites
as an outout it returns the min, max, mean and median of the probabilites for all intervals in one task
##pred - vector of probability prediction
##Same_Task - find Same Task avtibity
##symp - symptom
"""
def make_intervals_class(pred,Same_Task,symp):
        
    prob_min = []; prob_max = []; prob_mean=[]; prob_median=[]; label_Task = [];
    for j in range(len(pred)):
       #print j
        index = np.where(Same_Task==Same_Task.as_matrix()[j])
        prob_min.append(np.min(pred[index])); prob_max.append(np.max(pred[index])); prob_mean.append(np.mean(pred[index])); prob_median.append(np.median(pred[index]));  
        label_Task.append(np.max(symp[index])) 
            
    prob_mat = np.stack((prob_min,prob_max, prob_mean,prob_median,label_Task),axis=1)      
    return prob_mat    


"""           
Check if the men is in Tagdata
"""
def CHECKin(num_of_patient,Tagdata,Subjectid):
    INDEX = np.zeros(np.shape(Tagdata)[0])
    for x in range(np.shape(Tagdata)[0]):
       if(x in Subjectid[num_of_patient][0]):
           INDEX[x] = 1
    return INDEX

