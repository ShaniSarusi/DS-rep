#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 07:06:33 2017

@author: HealthLOB
"""
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

input_signal = Input((126,2))
#
# our data is 1d. however, 1d doesn't work well in this package, so, we do 2d and set the the second dimension to 1.
x = ZeroPadding1D(1)(input_signal)  # this means you run it on signal
x = Conv1D(32, 3, activation='relu', padding='same')(x)  # this means you run it on x, and so forth in the next lines.
x = MaxPooling1D(2)(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(4, 3, activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(64, activation='tanh')(x)
# at this point the representation is (8, 4, 4) i.e. 128-dimensional
# x = Dense(64, activation='relu')(encoded)
x = Reshape((32, 2))(encoded)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = UpSampling1D((2))(x)
decoded = Conv1D(32, 3, activation='relu', padding='same')(x)
decoded = Conv1D(2, 3, activation='sigmoid')(decoded)

#output_layer = Dense(32, activation='relu')(encoded)
output_layer = Dense(16, activation='relu')(encoded)
output_layer = Dense(8, activation='relu')(output_layer)
output_layer = Dense(1, activation='sigmoid')(output_layer)
        
autoencoder = Model(input_signal,decoded)
optimizer = optimizers.Adam(lr=0.00001)
# Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#


encoder = Model(input_signal, encoded)
encoder.compile(optimizer='adam', loss='binary_crossentropy')
#

data_here = data_for_model_raw_normlize.copy() 
number_of_data = np.shape(data_here)[len(np.shape(data_here))-1]
test = random.sample(range(np.shape(data_here)[0]), 10000)
train = list(set(range(np.shape(data_here)[0])) - set(test))
xtest = data_here[test]
#xtest = np.reshape(xtest,(np.shape(xtest)[0],np.shape(xtest)[1],number_of_data,1))
xtrain = data_here[train]
#xtrain = np.reshape(xtrain,(np.shape(xtrain)[0],np.shape(xtrain)[1],number_of_data))
# the first xtrain is the input, the second is the output. They are the same, since this is an auto-encoder.
autoencoder.fit(xtrain, xtrain,epochs=3 , batch_size=128, shuffle=True, validation_data=(xtest, xtest))

Dys_det = Model(input_signal ,output_layer)
Dys_det.compile(optimizer=optimizer, loss='binary_crossentropy')
Wsave = Dys_det.get_weights()

for layer in autoencoder.layers:
    layer.trainable = False
    
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
encoder.compile(optimizer=optimizer, loss='binary_crossentropy')
Dys_det.compile(optimizer=optimizer, loss='binary_crossentropy')


TagLow =  np.stack( np.stack((np.apply_along_axis(normlize_sig,1,TagverDenoise),np.apply_along_axis(normlize_sig,1,TaghorDenoise) )),axis=2)
#TagLow = np.stack((TagVerFilter, TagHorFilter),axis=2)

pred_ALL = []
##is the sample in the cluster of the right activity 
clus = [5]
cond = np.asarray(lmap(lambda x: x in clus, Task))
symp =Dys.copy()

logo = LeaveOneGroupOut()
cv = logo.split(TagLow[cond==True], symp[cond==True], meta.SubjectId[cond==True])
cv1 = list(cv)
cv = list(cv1)

def my_dys_func(learn_rate,beta_1,beta_2,output_layer = output_layer,input_layer= input_signal):
    Dys_det = Model(input_signal ,output_layer)
    optimizer = Adam(lr=learn_rate, beta_1=beta_1,beta_2 = beta_2)
    Dys_det.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return Dys_det

kfold = StratifiedKFold(n_splits=cv , shuffle=True, random_state=1234)
blabla = kfold.split(TagLow[cond==True],symp[cond==True])

batch_size= [32,64,128,256]
learn_rate = [0.0001,0.0001,0.001, 0.01, 0.1, 0.2, 0.3]
beta_1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
beta_2 = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

model_sklearn = KerasClassifier(build_fn=my_dys_func,epochs=20)
param_grid = dict(batch_size = batch_size,learn_rate=learn_rate, beta_1=beta_1,beta_2 = beta_2)
grid = GridSearchCV(estimator=model_sklearn, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(TagLow[cond==True],symp[cond==True])
#results = cross_val_score(model_sklearn ,TagLow[cond == True], symp[cond==True], cv=list(cv))

class_weight = {0 : 1,  1: 1.5}
for train, test in cv:
    print(test)
    #Dys_model = KerasClassifier(build_fn=my_dys_func, epochs=150, batch_size=10, verbose=0)
    #Dys_model.fit(TagLow[cond==True][train], symp[cond==True][train],epochs=15 , batch_size=128, validation_data=(TagLow[cond==True][test], symp[cond==True][test]),verbose=0)
    Dys_det.set_weights(Wsave)
    Dys_det.fit(TagLow[cond==True][train], symp[cond==True][train],epochs=20 , batch_size=64, validation_data=(TagLow[cond==True][test], symp[cond==True][test]),verbose=2,class_weight = class_weight)#,class_weight = class_weight
    pred_ALL.append(Dys_det.predict(TagLow[cond==True][test])[:,0])

########
import matplotlib.pyplot as plt
plt.hist(Dys_det.predict(TagLow[cond==True]))
##############################3
input_signal = Input(shape=(250,2))

    # our data is 1d. however, 1d doesn't work well in this package, so, we do 2d and set the the second dimension to 1.
x = ZeroPadding1D((3))(input_signal)  #this means you run it on signal
#x = Conv1D(128,2, activation='relu', padding='same')(x)  #this means you run it on x, and so forth in the next lines.
x = Conv1D(64,2, activation='relu',padding='same')(x)
x = MaxPooling1D((2))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((4))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((2))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((2))(x)
x = Flatten()(x)
encoded = Dense(32, activation='relu')(x)

x = Dense(126, activation='relu')(encoded)
x = Reshape((63,2))(x)
x = UpSampling1D((2))(x)
x = Conv1D(16,2, activation='relu',padding='same')(x)
x = Conv1D(16,2, activation='relu',padding='same')(x)
x = UpSampling1D((2))(x)
decoded = Conv1D(2,3, activation='linear')(x)    

autoencoder = Model(input_signal, decoded)
#Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoder = Model(inputs= input_signal, outputs=encoded)
encoder.compile(optimizer='adam', loss='binary_crossentropy')


#######################3
"""
pred_ALL = []
##is the sample in the cluster of the right activity 
clus = [1,2]
cond = np.asarray(lmap(lambda x: x in clus, Task))
symp = Dys.copy()
space = {'batch_size': hp.choice('batch_size',range(32,256)),'lr': hp.uniform('lr',0,1),'beta_1': hp.uniform('beta1',0,1),'beta_2': hp.uniform('beta2',0,1),'epsilon': hp.uniform('epsilon',0,0.001),'decay': hp.uniform('decay',0,0.001)}

logo = LeaveOneGroupOut()
cv = logo.split(TagLow[cond==True], symp[cond==True], meta.SubjectId[cond==True])
cv1 = list(cv)
cv = list(cv1)
model_sklearn = KerasClassifier(build_fn=my_dys_func,epochs=10 )

hyper_opt_keras = BayesianHyperOpt(space = space,estimator=model_sklearn(), scoring=None ,cv =  3,max_evals = 10)
"""
########################################
input_signal = Input(shape=(126,))

x = Reshape((63,2))(input_signal)
x = UpSampling1D((2))(x)
x = Conv1D(16,2, activation='relu',padding='same')(x)
x = Conv1D(16,2, activation='relu',padding='same')(x)
x = UpSampling1D((2))(x)
genrator = Conv1D(2,3, activation='linear')(x)    

input_disc = Input(shape=(250,2))

x = ZeroPadding1D((3))(input_disc)  #this means you run it on signal
x = Conv1D(128,2, activation='relu', padding='same')(x)  #this means you run it on x, and so forth in the next lines.
x = Conv1D(64,2, activation='relu',padding='same')(x)
x = MaxPooling1D((2))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((4))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((2))(x)
x = Conv1D(32,2, activation='relu', padding='same')(x)
x = MaxPooling1D((2))(x)
x = Flatten()(x)
disc= Dense(32, activation='relu')(x)
disc= Dense(16, activation='relu')(disc)

disc= Dense(1, activation='relu')(disc)

gans_gen = Model(input_signal, genrator)
gans_gen.compile(loss='binary_crossentropy', optimizer='adam')

gans_disc = Model(input_disc, disc)
gans_disc.compile(loss='binary_crossentropy', optimizer='adam')

input_my_ass = Input(shape=(126,))
blabla = gans_gen(input_my_ass)

gan_V = gans_disc(blabla)

GAN = Model(input_my_ass, gan_V)
GAN.compile(loss='binary_crossentropy', optimizer='adam')

BACH_SIZE = 2000
Y_fake1 = np.reshape(np.random.normal(0,1,BACH_SIZE*250),(BACH_SIZE,250))
Y_fake2 = np.reshape(np.random.normal(0,1,BACH_SIZE*250),(BACH_SIZE,250))
Y_fake = np.stack((np.apply_along_axis(normlize_sig,1,Y_fake1),np.apply_along_axis(normlize_sig,1,Y_fake2) ),axis=2)
data_for_gans = np.vstack((Y_fake,data_here[np.random.randint(0,len(data_here),size=BACH_SIZE)]))

Y_label = np.hstack((np.zeros(BACH_SIZE),np.ones(BACH_SIZE)))
Y_label = lmap(lambda x: int(x),Y_label)

gans_disc.fit(data_for_gans,  np.asarray(Y_label),epochs=3 , batch_size=32, shuffle=True)
optimizer = optimizers.Adam(lr=0.00001)
for count in range(1000):
    print(count)    
    for layer in gans_disc.layers:
        layer.trainable = True
        
    gans_gen.compile(loss='binary_crossentropy', optimizer=optimizer)
    gans_disc.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
     
        
    BACH_SIZE = 128
    Y_fake1 = np.reshape(np.random.normal(0,1,BACH_SIZE*250),(BACH_SIZE,250))
    Y_fake2 = np.reshape(np.random.normal(0,1,BACH_SIZE*250),(BACH_SIZE,250))
    Y_fake = np.stack((np.apply_along_axis(normlize_sig,1,Y_fake1),np.apply_along_axis(normlize_sig,1,Y_fake2) ),axis=2)
    data_for_gans = np.vstack((Y_fake,data_here[np.random.randint(0,len(data_here),size=BACH_SIZE)]))
    
    Y_label = np.hstack((np.zeros(BACH_SIZE),np.ones(BACH_SIZE)))
    Y_label = lmap(lambda x: int(x),Y_label)
    
    d_loss = gans_disc.train_on_batch(data_for_gans,  np.asarray(Y_label))
    print(d_loss)
    for layer in gans_disc.layers:
        layer.trainable = False
    
    gans_gen.compile(loss='binary_crossentropy', optimizer=optimizer)
    gans_disc.compile(loss='binary_crossentropy', optimizer=optimizer)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    Y_fake1 = np.reshape(np.random.normal(0,1,BACH_SIZE*126),(BACH_SIZE,126))
      
    Y_label = np.hstack((np.zeros(BACH_SIZE)))
    Y_label = lmap(lambda x: int(x),Y_label)
    
    g_loss = GAN.train_on_batch(Y_fake1 ,  np.asarray(Y_label))
    print(g_loss)
    


