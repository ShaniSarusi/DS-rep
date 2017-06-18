# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:11:47 2016

@author: awagner
"""

#from cvxpy import *

"""
##Denoising of the data with wavelet
"""
def denoise(data):
    WC = pywt.wavedec(data,'sym8')
    threshold=0.0045*np.sqrt(2*np.log2(256))
    NWC = lmap(lambda x: pywt.threshold(x,threshold,'soft'), WC)
    return pywt.waverec( NWC, 'sym8')
"""
##Wavelet to array
"""
def toarraywav(data):
    WC = pywt.wavedec(data,'sym8')
    return np.array((list(chain.from_iterable(WC))))
"""
##Denoise the data with wavelet and butter filter
"""
def denoise2(data):
    if np.std(data)<0.01:
        result = denoise(data)
    else:
        result = butter_bandpass_filter(data - np.mean(data), 0.2, 3.5, 50, order=4)
    return  result
"""
##Fused lasso for data denoiseing
"""
def fusedlasso(sig,beta,mymatrix):   
    sig = np.reshape(sig,250)
    x = Variable(len(sig))
    #if np.std(sig)<0.05:
    #obj = Minimize(square(norm(x-sig))+tv(mul_elemwise(beta,x)))
    obj = Minimize(square(norm(x-sig))+beta*quad_form(x,mymatrix))
    prob = Problem(obj)
    prob.solve()  # Returns the optimal value.
    res = x.value
    res = np.asarray(res.flatten())
    return res[0]
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
Projection of the data to 2 dimension         
"""
def projGrav(XYZ):
    XYZ = np.reshape(XYZ,(int(len(XYZ)/3),3))
    ver = []; hor = []
    G = [np.mean(XYZ[:,0]),np.mean(XYZ[:,1]),np.mean(XYZ[:,2])]
    G_norm = G/np.sqrt(sum(np.power(G,2)))
    for i in range(len(XYZ[:,0])):
        ver.append(np.dot([XYZ[i,:]],G))
        hor.append(np.sqrt(np.dot(XYZ[i,:]-ver[i]*G_norm,XYZ[i,:]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver),len(ver))
    return np.asarray(ver), np.asarray(hor)

##Normlize the signal    
def normlize_sig(sig):
    y = (sig  - np.min(sig))/(np.max(sig)-np.min(sig))
    return y

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
        

