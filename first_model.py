# import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json, Model
from keras.layers import Dense, Activation, Conv2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from keras.utils import np_utils
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger


randomSeed = 42

# Function to convert the the class from binray to decimal
def decimalDecoding(inputdata):
    decoded_data = []
    for i in range(inputdata.shape[0]):
        decoded_data.append(np.argmax(inputdata[i]))
    return np.array(decoded_data)


inputShape = (224, 224, 3)
inputImageShape = (224, 224, 3)
epoch = 10
btachSize = 8
num__of_output_classes = 2
random.seed(randomSeed)
learningRate = 0.01
data = []
labels = []


data_pickle = open('E:\\data\\BUE\\Year 3\\Semster one\\GP\\Nour\\Main\\Nour.pkl', 'rb')
[dataX, labels] = pickle.load(data_pickle)
data_pickle.close()

label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_label = label_encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_label)

# split data into train and test with ratio 90,10 respectively
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, train_size=0.90, random_state=42)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, train_size=0.9)

training = False
if training:
    
    input_images = Input(shape = (224, 224, 3))
    x1 = Conv2D(32, (3, 3), strides = (1, 1), padding = 'same')(input_images)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size = (2, 2))(x1)
    x2 = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D(pool_size = (2, 2))(x2)
    x3 = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D(pool_size = (2, 2))(x3)
    x4 = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same')(x3)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = MaxPooling2D(pool_size = (2, 2))(x4)
    x5 = Flatten()(x4)
    x6 = Dense(500, activation = 'relu')(x5)
    outputs = Dense(2, activation = 'sigmoid')(x6)
    
    model = Model(inputs = input_images, outputs = outputs)
    
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epoch, batch_size=btachSize)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_00.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_00.h5")




