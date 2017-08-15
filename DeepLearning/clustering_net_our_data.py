#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 03:24:45 2017

@author: HealthLOB
"""

input_signal = Input((250,3))#augment_or_not
#
#x = ZeroPadding1D(3)(input_signal) 
x = Conv1D(32, 32, activation='relu')(input_signal) 
x = MaxPooling1D(2)(x)
x = Dropout(0.25)(x)
x = Conv1D(16, 16, activation='relu')(x)
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

Input_for_cluster =  Input((32,))
#ActiveLayer = Dropout(0.2)(OneForActandSymp)
ActiveLayer= Dense(32, activation='tanh')(OneForActandSymp)
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
autoencoder_layer = Conv1D(4,4, activation='tanh')(autoencoder_layer)
autoencoder_layer = UpSampling1D(2)(autoencoder_layer)
autoencoder_layer = Conv1D(3,1, activation='linear')(autoencoder_layer)
##Here is the final symtpoms
sympLayer = Dense(16, activation='tanh')(ActiveLayer)
sympLayer  = BatchNormalization()(sympLayer )
close_to_output = Dense(16, activation='tanh')(sympLayer )
#sympLayer = BatchNormalization()(sympLayer)

close_to_output = Dense(16, activation='tanh')(close_to_output)

input_home_or_not = Input((1,))
#attention_probs = Dense(16, activation='softmax', name='attention_vec')(close_to_output)
#attention_mul = Multiply((close_to_output, attention_probs), output_shape=32, name='attention_mul')
End_point = Dense(1, activation='sigmoid')(close_to_output)

symp_cluster = Model([input_signal, input_home_or_not], [End_point, autoencoder_layer, ActiveLayer])
feature_extract = Model(input_signal, sympLayer)

#active_class = Model(input_signal, End_point_Active)

optimizer = optimizers.adam(lr = 0.0001)

def penalized_loss(fake_or_not):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred-y_true)*fake_or_not,axis = -1)
    return loss

def cluster_function(cluster_center):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred-y_true),axis = -1)
    return loss


symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse','mse'],metrics=['accuracy'], loss_weights=[1.,0.5,0.])
feature_extract.compile(optimizer=optimizer, loss = 'mse',metrics=['accuracy'])

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
    optimizer = optimizers.adam(lr = 0.0001)

    symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse','mse'],metrics=['accuracy'], loss_weights=[1.,0.5,0.])


    xtest = TagLow[test]
    xtrain = TagLow[train]
    
    MyCenters = np.reshape(np.random.normal(0,2,32*len(labels_for_deep)),[len(labels_for_deep),32])

    symp_cluster.fit([xtrain, home_or_not[train]], [augment_dys[train],xtrain,MyCenters[train]],epochs=1 , batch_size = 128, shuffle=True,verbose=2)#
    cluster_features = symp_cluster.predict([xtrain, home_or_not[train]])[2]
    cluster_labels = KMeans(n_clusters=10).fit(cluster_features)
    MyCenters = [cluster_labels.cluster_centers_[j] for j in cluster_labels.labels_]
    MyCenters = np.vstack(MyCenters)
    symp_cluster.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mae','mse'],
                                                    metrics=['accuracy'], loss_weights=[1.,1.,1.])
    
    best_score = 0
    for i in range(1,4):
        print(i)
        if(i==2):
            K.set_value(symp_cluster.optimizer.lr,0.00005)
        if(i==3):
            K.set_value(symp_cluster.optimizer.lr,0.00001)
        symp_cluster.fit([xtrain, home_or_not[train]], [augment_dys[train],xtrain,MyCenters],epochs=i , batch_size = 128, 
                         shuffle=True,verbose=2, validation_data=([xtest[augment_or_not[test] == 1],
                         home_or_not[test][augment_or_not[test] == 1]], [augment_dys[test][augment_or_not[test] == 1],
                         xtest[augment_or_not[test] == 1],MyCenters[range(len(test))][augment_or_not[test] == 1]]))
        cluster_features =  symp_cluster.predict([xtrain, home_or_not[train]])
        cluster_labels = KMeans(n_clusters=10, random_state = 1234).fit(cluster_features[2])
        MyCenters = [cluster_labels.cluster_centers_[j] for j in cluster_labels.labels_]
        MyCenters = np.vstack(MyCenters)
    temp_res = symp_cluster.predict([xtest[augment_or_not[test] == 1], np.ones(len(test))])
    res.append(temp_res[0])
    symp_cor_res.append(augment_dys[test][augment_or_not[test] == 1])
    order_final.append(augment_task_ids[test][augment_or_not[test] == 1])
    features_from_deep.append(feature_extract.predict(xtest[augment_or_not[test] == 1]))
    print(confusion_matrix(symp_cor_res[len(symp_cor_res)-1],np.where(temp_res[0]>0.5,1,0)))
    
feature_deep1 = np.vstack(features_from_deep)
feature_deep1 = np.column_stack((np.hstack(order_final), feature_deep1))
feature_deep2 = feature_deep1[feature_deep1[:,0].argsort()]

all_pred['prediction_probability'] = np.vstack(res)
all_pred['true_label'] = np.vstack(symp_cor_res)
all_pred['task'] = np.hstack(order_final)
all_pred['patient'] = np.hstack(order_final)%3