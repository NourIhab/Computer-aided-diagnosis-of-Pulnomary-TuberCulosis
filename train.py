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
from loaddata import load_data
from cnn_model import cnn_model_structure

def train_model():
    # some declared variables
    randomSeed = 42
    networkInitialize = glorot_normal()
    inputImageShape = (224, 224, 3)
    epoch = 200
    btachSize = 32
    num_of_output_classes = 2
    random.seed(randomSeed)
    learningRate = 0.01


    trainX, testX, trainY, testY = load_data()
    # augmentation process
    augmentaion = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                     horizontal_flip=True, fill_mode="nearest")

    checkpoint = ModelCheckpoint('models\\model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto')
    csv_logger = CSVLogger('report\\log_' + str(learningRate) + '.csv', append=False, separator=';')
    # training
    # compile the model
    model = cnn_model_structure(input_shape=inputImageShape,num_classes=num_of_output_classes)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    # print(model.summary())
    model = model.fit_generator(augmentaion.flow(trainX, trainY, batch_size=btachSize),
                                validation_data=(testX, testY), steps_per_epoch=len(trainX),
                                epochs=epoch, callbacks=[csv_logger, checkpoint])

