# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:30:10 2017

@author: awagner
"""
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from sklearn.cluster import KMeans
from keras import optimizers
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_train1 = x_train.reshape([60000,28**2])
x_test1 = x_test.reshape([10000, 28**2])  # adapt this if using `channels_first` image data format

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K


def loss_cluster(y_true,y_pred):
    f = K.constant(0,dtype=tf.float32)
    
    for i in range(32):
        print(i)
        for j in range(i):
            div_y_pred = K.sum(K.square(y_pred[i,:] - y_pred[j,:]))
            step1 = tf.cond(div_y_pred<0.1,lambda: K.sum(K.exp(-0.5*div_y_pred)*K.abs(y_pred[i,:] - y_pred[j,:])),lambda: tf.constant(0, dtype=tf.float32))
            f = step1 + f
    return f 


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)

encoded = Dense(16, activation='tanh')(x)
#encoded = Dense(2, activation='tanh')(x)
x = Dense(64, activation='tanh')(encoded)

x = Reshape((4,4,4))(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

optimizer = optimizers.adam(lr = 0.001)
autoencoder = Model(input_img, [decoded,encoded])
autoencoder.compile(optimizer=optimizer, loss=['binary_crossentropy',loss_cluster], loss_weights=[1., 0.0])

encoder = Model(input_img, encoded)
encoder.compile(optimizer='adam', loss='binary_crossentropy')# , loss_weights=[1., 0.])

    
autoencoder.fit(x_train, [x_train,x_train1],
                epochs=50,
                batch_size=20,
                shuffle=True)

Y = autoencoder.predict(x_test)
plt.scatter(Y[1][:,0], Y[1][:,1] ,c = y_test)

result_New = KMeans(n_clusters=10).fit_predict(Y[1])

metrics.adjusted_mutual_info_score(y_test ,result_New)
metrics.v_measure_score(y_test ,result_New)
result_New1 = KMeans(n_clusters=10).fit_predict(x_test1)

############################################################################
