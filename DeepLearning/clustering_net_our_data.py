#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 03:24:45 2017

@author: HealthLOB
"""

input_signal = Input((250,3))
#
#x = ZeroPadding1D(3)(input_signal) 
x = Conv1D(32, 32, activation='relu', padding='same')(input_signal) 
x = MaxPooling1D(2)(x)
x = Dropout(0.25)(x)
x = Conv1D(16, 16, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
x = Conv1D(4, 4, activation='relu', padding='same')(x)
#BatchNormalization()(x)
x = MaxPooling1D(4)(x)
x = Conv1D(4, 3, activation='relu', padding='same')(x)
x = MaxPooling1D(2)(x)
#x = GRU(32,  activation='tanh', return_sequences=True)(x)
#attention_mul = attention_3d_block(x)
#x = Flatten()(attention_mul)
x = Flatten()(x)
OneForActandSymp = Dense(32, activation='relu')(x)#,activity_regularizer=regularizers.l2(0.001)
#OneForActandSymp = BatchNormalization()(OneForActandSymp)

Input_for_cluster =  Input((16,))
##Here is activity
ActiveLayer = Dropout(0.2)(OneForActandSymp)
ActiveLayer= Dense(16, activation='linear')(ActiveLayer)
ActiveLayer = BatchNormalization()(ActiveLayer)
#ActiveLayer = Dense(16, activation='relu')(ActiveLayer)
#End_point_Active = Dense(4, activation='softmax')(ActiveLayer)

##Here is autoencoer
autoencoder_layer = Dense(32, activation='relu')(ActiveLayer)
autoencoder_layer = Reshape((16,2))(autoencoder_layer)
autoencoder_layer = UpSampling1D(2)(autoencoder_layer)
autoencoder_layer = Conv1D(4,4, activation='relu',padding='same')(autoencoder_layer)
autoencoder_layer = UpSampling1D(2)(autoencoder_layer)
autoencoder_layer = Conv1D(16,4, activation='relu',padding='same')(autoencoder_layer)
autoencoder_layer = UpSampling1D(2)(autoencoder_layer)
autoencoder_layer = Conv1D(4,4, activation='relu')(autoencoder_layer)
autoencoder_layer = UpSampling1D(2)(autoencoder_layer)
autoencoder_layer = Conv1D(3,1, activation='relu')(autoencoder_layer)
##Here is the final symtpoms
sympLayer = Dense(16, activation='relu')(ActiveLayer)
#sympLayer = BatchNormalization()(sympLayer)
close_to_output = Dense(16, activation='relu')(sympLayer)

input_home_or_not = Input((1,))
#attention_probs = Dense(16, activation='softmax', name='attention_vec')(close_to_output)
#attention_mul = Multiply((close_to_output, attention_probs), output_shape=32, name='attention_mul')
End_point = Dense(1, activation='sigmoid')(close_to_output)

symp_cluster = Model([input_signal, input_home_or_not], [End_point, autoencoder_layer, ActiveLayer])
#active_class = Model(input_signal, End_point_Active)

optimizer = optimizers.adam(lr = 0.0005)

def penalized_loss(fake_or_not):
    def loss(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_pred,y_true)*fake_or_not,axis = -1)
    return loss

def cluster_function(cluster_center):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred-y_true),axis = -1)
    return loss


symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse','mse'],metrics=['accuracy'], loss_weights=[1.,0.5,0.])

weights_start_symp = symp_cluster.get_weights()


from sklearn.cluster import KMeans

logo = LeaveOneGroupOut()
cv = logo.split(TagLow, augment_dys, labels_for_deep)
cv1 = list(cv)
cv = list(cv1)

order_final = []
res = []
symp_cor_res = []
features_from_deep = []
for train, test in cv:
    
    
    symp_cluster.set_weights(weights_start_symp)
    
    symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse','mse'],metrics=['accuracy'], loss_weights=[1.,0.5,0.])

    Mytrain = train[range(len(train) - len(train)%deep_params['batch_size'])]
    Mytest = test[range(len(test) - len(test)%deep_params['batch_size'])]

    xtest = TagLow[test]
    xtrain = TagLow[Mytrain]
    
    MyCenters = np.reshape(np.random.normal(0,2,16*len(labels_for_deep)),[len(labels_for_deep),16])

    symp_cluster.fit([xtrain, home_or_not[Mytrain]], [augment_dys[Mytrain],xtrain,MyCenters[Mytrain]],epochs=1 , batch_size = 128, shuffle=True,verbose=2)#
    cluster_features = symp_cluster.predict([xtrain, home_or_not[Mytrain]])[2]
    cluster_labels = KMeans(n_clusters=5).fit(cluster_features)
    MyCenters = [cluster_labels.cluster_centers_[j] for j in cluster_labels.labels_]
    MyCenters = np.vstack(MyCenters)
    symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse',
             cluster_function(cluster_center = Input_for_cluster)],metrics=['accuracy'], loss_weights=[1.,0.1,0.25])

    for i in range(1,7):
        print(i)
        if(i==3):
            K.set_value(symp_cluster.optimizer.lr,0.0001)
        if(i==20):
            K.set_value(symp_cluster.optimizer.lr,0.00005)
        symp_cluster.fit([xtrain, home_or_not[Mytrain]], [augment_dys[Mytrain],xtrain,MyCenters],epochs=i , batch_size = 128, shuffle=True,verbose=2, validation_data=([xtest[augment_or_not[test] == 1],home_or_not[test][augment_or_not[test] == 1]], [augment_dys[test][augment_or_not[test] == 1],xtest[augment_or_not[test] == 1],MyCenters[range(len(test))][augment_or_not[test] == 1]]))
        cluster_features =  symp_cluster.predict([xtrain, home_or_not[Mytrain]])
        cluster_labels = KMeans(n_clusters=5, random_state = 1234).fit(cluster_features[2])
        MyCenters = [cluster_labels.cluster_centers_[j] for j in cluster_labels.labels_]
        MyCenters = np.vstack(MyCenters)
    temp_res = symp_cluster.predict([xtest[augment_or_not[test] == 1], np.ones(len(test))])
    res.append(temp_res[0])
    symp_cor_res.append(augment_dys[test][augment_or_not[test] == 1])
    order_final.append(augment_task_ids[test][augment_or_not[test] == 1])
    print(confusion_matrix(symp_cor_res[len(symp_cor_res)-1],np.where(temp_res[0]>0.5,1,0)))