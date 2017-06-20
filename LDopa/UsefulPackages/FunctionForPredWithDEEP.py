# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:11:47 2016

@author: awagner
"""

#from cvxpy import *


"""
Run your deep learning net with test and train  
"""
def runnetwork(netName, sample_size,data,lost = 'binary_crossentropy',num_of_epochs = 7, num_of_batch_size=256):        
    decoded, encoded = netName(np.shape(data)[1],lost = lost)
    number_of_data = np.shape(data)[len(np.shape(data))-1]
    test = random.sample(range(np.shape(data)[0]), sample_size)
    train = list(set(range(np.shape(data)[0])) - set(test))
    xtest = data[test]
    #xtest = np.reshape(xtest,(np.shape(xtest)[0],np.shape(xtest)[1],number_of_data,1))
    xtrain = data[train]
    #xtrain = np.reshape(xtrain,(np.shape(xtrain)[0],np.shape(xtrain)[1],number_of_data))
    # the first xtrain is the input, the second is the output. They are the same, since this is an auto-encoder.
    decoded.fit(xtrain, xtrain,epochs=num_of_epochs , batch_size=num_of_batch_size, shuffle=True, validation_data=(xtest, xtest),verbose=2)
    return decoded, encoded
 
      
def runnetLSTM(netName, sample_size,data):        
    decoded, encoded = netName(len(data[0]),11,lost = 'binary_crossentropy')
    number_of_data = np.shape(data)[len(np.shape(data))-1]
    test = random.sample(range(np.shape(data)[0]), sample_size)
    train = list(set(range(np.shape(data)[0])) - set(test))
    xtest = data[test]
    xtest = np.reshape(xtest,(np.shape(xtest)[0],np.shape(xtest)[1],number_of_data))
    xtrain = data[train]
    xtrain = np.reshape(xtrain,(np.shape(xtrain)[0],np.shape(xtrain)[1],number_of_data))
    # the first xtrain is the input, the second is the output. They are the same, since this is an auto-encoder.
    decoded.fit(xtrain, xtrain,nb_epoch=3, batch_size=128, shuffle=True, validation_data=(xtest, xtest))
    return decoded, encoded




"""
Select feature
"""
def selectfeature(data,symp,clus,Task_for_model,C=1,method='Lasso'):
    cond = np.asarray(lmap(lambda x: x in clus, Task_for_model))
    if method == 'Lasso':        
         clf = LogisticRegression(penalty='l1',C=C)
         clf.fit(data[cond==True],symp[cond==True])
         return clf.coef_
        
    if method == 'Forest':
         clf = RandomForestClassifier(n_estimators=100,max_depth=20)
         clf.fit(data[cond==True],symp[cond==True])
         return clf.feature_importances_
        

