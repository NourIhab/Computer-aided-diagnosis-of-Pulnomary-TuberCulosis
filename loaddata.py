import numpy as np
import pickle
import cv2
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data():
    data_pickle = open('Nour.pkl', 'rb')
    [dataX, labels] = pickle.load(data_pickle)
    data_pickle.close()

    # encoding the data into one hot encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_label = label_encoder.transform(labels)
    dataY = np_utils.to_categorical(encoded_label)

    # split data into train and test with ratio 90,10 respectively
    (trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)
    return trainX, testX, trainY, testY