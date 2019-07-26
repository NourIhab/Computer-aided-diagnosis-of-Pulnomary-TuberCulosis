#import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
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
inputShape = (224, 224, 3)
epoch=200
BS = 32
num_classes = 2
random.seed(randomSeed)
INIT_LR=0.01

data = []
labels = []

def classification_report_csv(report,directory):
    report_data = [] # array to hold the data
    lines = report.split('\n')
    for line in lines[2:-3]: # hya el function  asln batseeeb etnen fel awel w talta fel a5er fa b3mel cut le awel etnen lines w a5er talta 3ashan me3mlesh exeption
        try:
            row = {}
            row_data_ = line.split('      ')
            row_data=[x for x in row_data_ if x]
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        except:
            pass
    dataframe = pd.DataFrame.from_dict(report_data) # inside panda there is function called dataframe
    dataframe.to_csv(directory+'_classification_report.csv', index = False) # index = false  for not putting index in the excel sheet

# to convert the the class from decimal (hot encoding) into binary
def decode(data):
    decoded_data = []
    for i in range(data.shape[0]):
        decoded_data.append(np.argmax(data[i]))
    return np.array(decoded_data)



# read from the saved  pickle file
data_fid = open('Nour_pre.pkl', 'rb')
[dataX, labels] = pickle.load(data_fid)
data_fid.close()

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)


dataY = np_utils.to_categorical(encoded_Y)
# split data into train and test with ratio 90,10 respectively
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)

#augmentation process
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Convoultional neural network structure
training = False # training is false to test not to train
if training: # if training = true strat train
    
    images = Input(shape=(224, 224, 3))
    x1 = Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize, padding='same')(images)
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
    x5= Activation('relu')(x5)
    x5 = MaxPooling2D(pool_size=(2, 2))(x5)
    x6= Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer=networkInitialize,  padding='same')(x5)
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
    
    model = Model(inputs=images, outputs=outputs)
    checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_acc', save_best_only=True, mode='auto') # to monitor the validation accuracy to save the best model
    csv_logger = CSVLogger('report\\log_'+str(INIT_LR)+'.csv', append=False, separator=';')

    sgd = optimizers.SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    print(model.summary())

	# serialize model to JSON
    model_json = model.to_json()
    with open('model.json', "w") as json_file:
        json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights('model_' + str(epoch) + '.h5')

    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX),
                            epochs=epoch, callbacks=[csv_logger, checkpoint])
    '''
    # plot the training loss and accuracy
    N = np.arange(0, epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy " + str(INIT_LR))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss\\Accuracy")
    plt.legend()
    plt.savefig('Plots\\history_fig')
    '''
else:

    # load json model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('model-100-0.998283-0.746269.h5')


    predictions = model.predict(testX)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1))
    print(report)
    classification_report_csv(report, 'test\\' + str(INIT_LR))


result = decode(model.predict(testX))
print(result)
ref_result = decode(testY)
print(ref_result)
acc = 100 * (1 - float(np.count_nonzero(result - ref_result))/float(len(result)))
print('Acc = ' + str(acc))
