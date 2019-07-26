import numpy as np
from keras.models import  Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.initializers import glorot_normal

networkInitialize = glorot_normal()

def cnn_model_structure(input_shape=(224, 224, 3),num_classes=2):

    input_Images = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(input_Images)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2))(x5)
    x6 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x5)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = MaxPooling2D(pool_size=(2, 2))(x6)
    x7 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(x6)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    x7 = MaxPooling2D(pool_size=(2, 2))(x7)
    x8 = Flatten()(x7)
    x9 = Dense(500, activation='relu')(x8)
    outputs = Dense(num_classes, activation='sigmoid')(x9)

    # creating the final model
    model = Model(inputs=input_Images, outputs=outputs)
    return model
