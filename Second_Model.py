#import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import random
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Flatten
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.initializers import glorot_normal

# some declared variables

randomSeed = 42
networkInitialize = glorot_normal()
inputDataShape = (224, 224, 3)
epoch = 50
BatchSize = 32
outputClasses = 2
random.seed(randomSeed)
learningRate = 0.0001


# Function to convert the the data from binray to decimal
def decimalDecoding(inputdata):
    decoded_data = []
    for i in range(inputdata.shape[0]):
        decoded_data.append(np.argmax(inputdata[i]))
    return np.array(decoded_data)


pickle_file = open('E:\\data\\BUE\\Year 3\\Semster one\\GP\\Nour\\Main\\Nour.pkl', 'rb') # read the binary data in the pickle file
[normalizedDatax, imageLabels] = pickle.load(pickle_file)
pickle_file.close() # close the pickle file

label_encoder = LabelEncoder()
label_encoder.fit(imageLabels)
encoded_label = label_encoder.transform(imageLabels)
dataY = np_utils.to_categorical(encoded_label)

# split data into train and test with ratio 90,10 respectively
(trainX, testX, trainY, testY) = train_test_split(normalizedDatax, dataY, test_size=0.10, random_state=42)
# split tarninig into valdiation and training with ratio 10, 90 respectively
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=42)


training = False
if training:

    InputImages = Input(shape=(224, 224, 3))
    x1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(InputImages)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    x2 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = Conv2D(64, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = Conv2D(128, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x4)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2))(x5)
    x6 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x5)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = MaxPooling2D(pool_size=(2, 2))(x6)
    x7 = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x6)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    x7 = MaxPooling2D(pool_size=(2, 2))(x7)
    x8 = Flatten()(x7)
    x9 = Dense(500,  activation='relu')(x8)
    outputs = Dense(2,  activation='sigmoid')(x9)
    
    model = Model(inputs=InputImages, outputs=outputs)

    stochastic_gradient_descent_optimizer = optimizers.SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epoch, batch_size=BatchSize)

    # serialize model to JSON
    model_json = model.to_json()
    with open('model_'+str(epoch)+'.json', "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights('model_'+str(epoch)+'.h5')

    

