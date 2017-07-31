# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from keras.layers import Embedding, merge, Input, Dropout, Dense, Conv2D, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, ZeroPadding1D, LSTM, RepeatVector
from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers 
from keras.callbacks import EarlyStopping 
from keras import backend as K
#cannot change border mode to same in maxpooling in theano. need to go to tensor flow.


def BuildCNNNet(input_size,lost):  # number of nodes in first layer. in this case 126.
    
    input_signal = Input(shape=(1,input_size,1))

    # our data is 1d. however, 1d doesn't work well in this package, so, we do 2d and set the the second dimension to 1.
    x = ZeroPadding2D((1,0))(input_signal)  #this means you run it on signal
    x = Convolution2D(64,4,1, activation='relu', border_mode='same')(x)  #this means you run it on x, and so forth in the next lines.
    x = MaxPooling2D((2,1))(x)
    x = Convolution2D(32,4,1, activation='relu', border_mode='same')(x)  #this means you run it on x, and so forth in the next lines.
    x = MaxPooling2D((2,1))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2,1))(x)
    x = Convolution2D(8,3,1, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2,1))(x) #this layer was called encoded and not x, since we will use it in the future (these are the features)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional

    x = Convolution2D(8,3,1, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(32,4,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(4,6,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    decoded = Convolution2D(3,3,1, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(1,3,1, activation='sigmoid')(decoded)  #this is the last layer. it is the same size as the input.

    autoencoder = Model(input_signal, decoded)
    #Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    autoencoder.compile(optimizer='adam', loss=lost)
    
    encoder = Model(input= input_signal, output=encoded)
    encoder.compile(optimizer='adam', loss=lost)
    
    return autoencoder, encoder  # returns the two different models, already compiled.
    
    

def BuildCNNNet2(input_size,lost):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input(shape=(1,input_size,2))
    #
    # our data is 1d. however, 1d doesn't work well in this package, so, we do 2d and set the the second dimension to 1.
    x = ZeroPadding2D((0,1))(input_signal)  #this means you run it on signal
    x = Convolution2D(32,3,1, activation='relu', border_mode='same')(x)  #this means you run it on x, and so forth in the next lines.
    x = MaxPooling2D((1,2))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Convolution2D(8,3,1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Convolution2D(4,3,1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((1,2))(x)
    x = Flatten()(x)
    encoded = Dense(64, activation='tanh')(x) 
    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    #x = Dense(64, activation='relu')(encoded)
    x = Reshape((2,32,1))(encoded)
    x = Convolution2D(8,3,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    decoded = Convolution2D(32,3,1, activation='relu', border_mode='same')(x)
    decoded = Convolution2D(2,3,1, activation='sigmoid')(decoded)  #this is the last layer. it is the same size as the input.
    autoencoder = Model(input_signal, decoded)
    #Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    autoencoder.compile(optimizer='adam', loss=lost)
    #
    encoder = Model(input= input_signal, output=encoded)
    encoder.compile(optimizer='adam', loss=lost)
    #
    return autoencoder, encoder  # returns the two different models, already compiled.
    
    
def BuildLSTM(input_size,num_of_signal,lost):
    
    inputs = Input(shape=(input_size, num_of_signal))
    encoded = LSTM(32)(inputs)

    decoded = RepeatVector(input_size)(encoded)
    decoded = LSTM(11, return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss=lost)
    encoder = Model(inputs, encoded)
    encoder.compile(optimizer='adam', loss=lost)
    return autoencoder, encoder
    
    
def BuildCNNNetRaw(input_size,lost):  # number of nodes in first layer. in this case 126.
    
    input_signal = Input(shape=(input_size,2))

    # our data is 1d. however, 1d doesn't work well in this package, so, we do 2d and set the the second dimension to 1.
    x = ZeroPadding1D((3))(input_signal)  #this means you run it on signal
    x = Conv1D(10,5, activation='linear', padding='same')(x)  #this means you run it on x, and so forth in the next lines.
    x = Conv1D(20,5, activation='relu',padding='same')(x)
    x = MaxPooling1D((2))(x)
    x = Conv1D(5,5, activation='relu', padding='same')(x)
    x = MaxPooling1D((4))(x)
    x = Conv1D(2,5, activation='relu', padding='same')(x)
    x = MaxPooling1D((2))(x)
    x = Conv1D(5,5, activation='relu', padding='same')(x)
    x = MaxPooling1D((2))(x)
    x = Flatten()(x)
    encoded = Dense(32, activation='relu')(x)

    x = Dense(126, activation='relu')(encoded)
    x = Reshape((63,2))(x)
    x = UpSampling1D((2))(x)
    x = Conv1D(5,5, activation='relu',padding='same')(x)
    x = UpSampling1D((2))(x)
    decoded = Conv1D(2,3, activation='linear')(x)    
    
    autoencoder = Model(input_signal, x)
    #Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    autoencoder.compile(optimizer='adam', loss=lost)
    
    encoder = Model(input= input_signal, output=encoded)
    encoder.compile(optimizer='adam', loss=lost)
    
    return autoencoder, encoder  # returns the two different models, already compiled.


def BuildCNNNet3(input_size, lost):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,2))
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
    decoded = Conv1D(2, 3, activation='sigmoid')(decoded)  # this is the last layer. it is the same size as the input.
    autoencoder = Model(input_signal,decoded)
    # Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    autoencoder.compile(optimizer='adam', loss=lost)
    #
    encoder = Model(input_signal, encoded)
    encoder.compile(optimizer='adam', loss=lost)
    #
    return autoencoder, encoder, encoded  # returns the two different models, already compiled.

def LSTMCNN(input_size, lost):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,2))
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
    x= LSTM(32)(x)
    #x = Flatten()(x)
    encoded = Dense(32, activation='tanh')(x)
    x = RepeatVector(32)(encoded )
    x= LSTM(2, return_sequences=True)(x)
    x = Reshape((32, 2))(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D((2))(x)
    decoded = Conv1D(32, 3, activation='relu', padding='same')(x)
    decoded = Conv1D(2, 3, activation='sigmoid')(decoded)  # this is the last layer. it is the same size as the input.
    autoencoder = Model(input_signal,decoded)
    # Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0)
    autoencoder.compile(optimizer='adam', loss=lost)
    #
    encoder = Model(input_signal, encoded)
    encoder.compile(optimizer='adam', loss=lost)
    #
    return autoencoder, encoder, encoded  # returns the two different models, already compiled.


def BuildCNNClassifirt(input_size, lost):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((250,3))
    #
    #x = ZeroPadding1D(3)(input_signal) 
    x = Conv1D(32, 16, activation='relu', padding='same')(input_signal) 
    x = MaxPooling1D(2)(x)
    #x = Dropout(0.5)(x)
    x = Conv1D(16, 8, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(8, 4, activation='relu', padding='same')(x)
    #BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(4, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    close_to_output = Dense(32, activation='relu')(x)#,activity_regularizer=regularizers.l2(0.001)
    BatchNormalization()(close_to_output)
    Dropout(0.5)(close_to_output)
    close_to_output = Dense(16, activation='relu')(close_to_output)
    BatchNormalization()(close_to_output)
    close_to_output = Dense(16, activation='relu')(close_to_output)
    End_point = Dense(1, activation='sigmoid')(close_to_output)
    encoder = Model(input_signal, End_point)
    optimizer = optimizers.adam(lr = 0.001)
    feature_extract = Model(input_signal,close_to_output)
    
    encoder.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    feature_extract.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    #
    return encoder, feature_extract

def BuildCNNClassWithActivity(input_size, lost, lr = 0.0001, loss_weights=[1., 0.01]):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,3))
    #
    #x = ZeroPadding1D(3)(input_signal) 
    x = Conv1D(32, 32, activation='relu')(input_signal) 
    x = MaxPooling1D(4)(x)
    #x = Dropout(0.25)(x)
    x = Conv1D(16, 16, activation='relu')(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(28, 4, activation='relu', padding='same')(x)
    #BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(28, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(4)(x)
    #x = GRU(32,  activation='tanh', return_sequences=True)(x)
    #attention_mul = attention_3d_block(x)
    #x = Flatten()(attention_mul)
    x = Flatten()(x)
    OneForActandSymp = Dense(64, activation='relu')(x)#
    #OneForActandSymp = BatchNormalization()(OneForActandSymp)
    
    #ActiveLayer = Dropout(0.5)(OneForActandSymp)
    ActiveLayer = Dense(32, activation='relu')(OneForActandSymp)
    ActiveLayer = BatchNormalization()(ActiveLayer)
    #ActiveLayer = Dense(16, activation='relu')(ActiveLayer)
    End_point_Active = Dense(6, activation='softmax')(ActiveLayer)
    
    
    #sympLayer=Dropout(0.5)(ActiveLayer)
    sympLayer = Dense(19, activation='relu')(ActiveLayer)
    sympLayer = BatchNormalization()(sympLayer)
    input_user = Input((20,))
    #Fuck_YOU = Embedding(19, 20)(input_user)
    Fuck_YOU = Dense(10)(input_user)
    Fuck_YOU = Dense(10)(Fuck_YOU)
    merging = layers.concatenate([sympLayer, Fuck_YOU])
    
    #
    close_to_output = Dense(19, activation='relu')(merging)
    close_to_output = Dense(8, activation='relu')(close_to_output)
    close_to_output = Dense(8, activation='tanh')(close_to_output)
    End_point = Dense(1, activation='sigmoid')(close_to_output)
    
    symp_class = Model([input_signal, input_user], [End_point, End_point_Active])
    #active_class = Model(input_signal, End_point_Active)
    
    optimizer = optimizers.adam(lr = lr)
    feature_extract = Model([input_signal , input_user],close_to_output)
    
    symp_class.compile(optimizer=optimizer, loss=[lost,'categorical_crossentropy'],metrics=['accuracy'], loss_weights=loss_weights)
    feature_extract.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    #
    return symp_class, feature_extract



def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, 7))(a)
    a = Dense(7, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def BuildCNNClassWithLSTM(input_size, lost, lr = 0.0001, loss_weights=[1., 0.1]):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,3))
    #
    #x = ZeroPadding1D(3)(input_signal) 
    x = Conv1D(32, 16, activation='relu', padding='same')(input_signal) 
    x = MaxPooling1D(2)(x)
    #x = Dropout(0.5)(x)
    x = Conv1D(16, 8, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = GRU(32,  activation='sigmoid', return_sequences=True)(x)
    attention_mul = attention_3d_block(x)
    x = Flatten()(attention_mul)
    #x = Flatten()(x)
    OneForActandSymp = Dense(32, activation='relu')(x)#,activity_regularizer=regularizers.l2(0.001)
    #OneForActandSymp = BatchNormalization()(OneForActandSymp)
    
    #ActiveLayer = Dropout(0)(OneForActandSymp)
    ActiveLayer= Dense(16, activation='relu')(OneForActandSymp)
    ActiveLayer = BatchNormalization()(ActiveLayer)
    #ActiveLayer = Dense(16, activation='relu')(ActiveLayer)
    End_point_Active = Dense(4, activation='softmax')(ActiveLayer)
    
    
    #sympLayer=Dropout(0.5)(OneForActandSymp)
    sympLayer = Dense(16, activation='relu')(ActiveLayer)
    #sympLayer = BatchNormalization()(sympLayer)
    close_to_output = Dense(16, activation='relu')(sympLayer)
    End_point = Dense(1, activation='sigmoid')(close_to_output)
    
    symp_class = Model(input_signal, [End_point, End_point_Active])
    #active_class = Model(input_signal, End_point_Active)
    
    optimizer = optimizers.adam(lr = lr)
    feature_extract = Model(input_signal,close_to_output)
    
    symp_class.compile(optimizer=optimizer, loss=[lost,'categorical_crossentropy'],
                       metrics=['accuracy'], loss_weights=loss_weights)
    feature_extract.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    #
    return symp_class, feature_extract


def BuildCNNClassWithAutoencoder(input_size, lost, lr = 0.001, loss_weights=[1.,0.8]):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,3))
    #
    #x = ZeroPadding1D(3)(input_signal) 
    x = Conv1D(32, 32, activation='relu', padding='same')(input_signal) 
    x = MaxPooling1D(2)(x)
    #x = Dropout(0.25)(x)
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
    
    ##Here is activity
    #ActiveLayer = Dropout(0.2)(OneForActandSymp)
    ActiveLayer= Dense(16, activation='relu')(OneForActandSymp)
    ActiveLayer = BatchNormalization()(ActiveLayer)
    #ActiveLayer = Dense(16, activation='relu')(ActiveLayer)
    #End_point_Active = Dense(4, activation='softmax')(ActiveLayer)
    
    ##Here is autoencoer
    autoencoder_layer = Dense(32, activation='relu')(OneForActandSymp)
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
    
    symp_class = Model([input_signal, input_home_or_not], [End_point, autoencoder_layer])
    #active_class = Model(input_signal, End_point_Active)
    
    optimizer = optimizers.adam(lr = lr)
    feature_extract = Model(input_signal,close_to_output)
    
    def penalized_loss(fake_or_not):
        def loss(y_true, y_pred):
            return K.mean(K.binary_crossentropy(y_pred,y_true)*fake_or_not,axis = -1)
        return loss

    symp_class.compile(optimizer=optimizer, loss=[penalized_loss(fake_or_not = input_home_or_not),'mse'],metrics=['accuracy'], loss_weights = loss_weights)
    feature_extract.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    #
    return symp_class, feature_extract


def MultiTaskNet(input_size, lost):  # number of nodes in first layer. in this case 126.
    #
    input_signal = Input((input_size,3))
    #
    #x = ZeroPadding1D(3)(input_signal) 
    x = Conv1D(32, 32, activation='relu', padding='same')(input_signal) 
    x = MaxPooling1D(2)(x)
    #x = Dropout(0.5)(x)
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
    
    ##Here is activity
    #ActiveLayer = Dropout(0.5)(OneForActandSymp)
    ActiveLayer= Dense(16, activation='relu')(OneForActandSymp)
    ActiveLayer = BatchNormalization()(ActiveLayer)
    ActiveLayer = Dense(16, activation='relu')(ActiveLayer)
    End_point_Active = Dense(4, activation='softmax')(ActiveLayer)
    

    ##Here is the final symtpoms
    sympLayer = Dense(16, activation='relu')(ActiveLayer)
    #sympLayer = BatchNormalization()(sympLayer)
    close_to_output = Dense(16, activation='relu')(sympLayer)
    
    #attention_probs = Dense(16, activation='softmax', name='attention_vec')(close_to_output)
    #attention_mul = merge([close_to_output, attention_probs], output_shape=32, name='attention_mul',mode = 'mul')
    End_point_Dys = Dense(2, activation='softmax')(close_to_output)
    End_point_trem = Dense(2, activation='softmax')(close_to_output)
    End_point_brady = Dense(2, activation='softmax')(close_to_output)
    
   
    def w_categorical_crossentropy(weights,y_pred_brad, lr = 0.0001):
        def myloss(y_true, y_pred):
            nb_cl = len(weights)
            final_mask = K.zeros_like(y_pred[:, 0])
            y_pred_max = K.max(y_pred, axis=1)
            y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
            y_pred_max_mat = K.equal(y_pred, y_pred_max)
            y_pred_max_brad = K.max(y_pred_brad, axis=1)
            y_pred_max_brad = K.reshape(y_pred_max_brad, (K.shape(y_pred_brad)[0], 1))
            y_pred_max_brad_mat = K.equal(y_pred_brad, y_pred_max_brad)
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += (K.cast(weights[c_t, c_p],tf.float32) * K.cast(y_pred_max_mat[:, c_p] ,tf.float32)* K.cast(y_pred_max_brad_mat[:, c_t],tf.float32))
            return K.categorical_crossentropy(y_pred, y_true) * final_mask
        return myloss

    weights = np.ones([2,2])
    weights[1,1] = 5
    symp_class = Model(input_signal, [End_point_Dys, End_point_trem, End_point_brady, End_point_Active])
    #active_class = Model(input_signal, End_point_Active)
    
    optimizer = optimizers.adam(lr = lr)
    feature_extract = Model(input_signal,close_to_output)
    
    symp_class.compile(optimizer=optimizer, loss=[w_categorical_crossentropy(weights = weights, y_pred_brad = End_point_trem), 'categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'], loss_weights=[1.,0.5,0.5,0.1])
    feature_extract.compile(optimizer=optimizer, loss=lost,metrics=['accuracy'])
    #
    return symp_class, feature_extract