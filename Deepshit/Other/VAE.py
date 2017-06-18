# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:07:58 2017

@author: awagner
"""
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size1, latent_dim))
    return z_mean + K.exp(z_log_var) * epsilon

def vae_loss(x1, x1_decoded_mean):
     x1 = K.flatten(x1)
     x1_decoded_mean = K.flatten(x1_decoded_mean)
     xent_loss = 250 * metrics.binary_crossentropy(x1, x1_decoded_mean)
     kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
     return xent_loss + kl_loss
     
     
def BuildCNNNetRaw(input_size,lost = vae_loss):
    
    batch_size1 = 100
    latent_dim = 2
    intermediate_dim = 32
    epsilon_std = 1.0
    epochs = 5
    
    input_layer = Input(batch_shape=(batch_size1,) + (2,250,1))
    x = Convolution2D(20,4,1, activation='relu', border_mode='same')(input_layer)
    x = MaxPooling2D((2,1))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2,1))(x)
    x = Convolution2D(16,3,1, activation='relu', border_mode='same')(x)
    flat = Flatten()(x)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    
    z = Lambda(sampling)([z_mean, z_log_var])
    
    encoder = Dense(intermediate_dim, activation='relu')(z)
    decoder_upsample = Dense(20*8*1, activation='relu')(encoder)
    output_shape = (batch_size1, 20, 8, 1)
    decoder_reshape = Reshape(output_shape[1:])(decoder_upsample)
    x = UpSampling2D((2,1))(decoder_reshape)
    x = Convolution2D(5,3,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(5,5,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(2,3,1, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2,1))(x)
    x = Convolution2D(2,3,1, activation='relu', border_mode='same')(x)
    
    vae = Model(input = input_layer, output = x)
    vae.compile(optimizer='adam', loss=lost)
    
     encoded = Model(input= input_layer, output=encoder)
     encoded.compile(optimizer='adam', loss=lost)
     
     return vae, encoded