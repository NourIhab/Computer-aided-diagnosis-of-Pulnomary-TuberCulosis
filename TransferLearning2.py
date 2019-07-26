from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Model, load_model , model_from_json
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
#import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import pandas as pd

Learning_rate = 0.001
num_classes = 2
batch_size = 16
EPOCHS = 200
output_dir = 'TransferLearning'
MODELS ={ "vgg16": VGG16, "vgg19": VGG19, "resnet": ResNet50}

def classification_report_csv(report,directory):
    data_report = []
    lines = report.split('\n')
    for line in lines[2:-3]:
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
    dataframe = pd.DataFrame.from_dict(data_report)
    dataframe.to_csv(directory+'_classification_report.csv', index = False)


# Function to convert the the data from binray to decimal
def decimalDecoding(inputdata):
    decoded_data = []
    for i in range(inputdata.shape[0]):
        decoded_data.append(np.argmax(inputdata[i]))
    return np.array(decoded_data)


data_pickle = open('E:\\data\\BUE\\Year 3\\Semster one\\GP\\Nour\\Main\\Nour.pkl', 'rb')
[dataX, labels] = pickle.load(data_pickle)
data_pickle.close()

#encoding the data into one hot encoding
label_encoder = LabelEncoder()
label_encoder.fit(labels)
encoded_label = label_encoder.transform(labels)
dataY = np_utils.to_categorical(encoded_label)

# split data into train and test with ratio 90,10 respectively
(trainX, testX, trainY, testY) = train_test_split(dataX, dataY, test_size=0.10, random_state=42)

#augmentation process
augmentaion = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

training = False
for Model_name in MODELS.keys():
    if training:

        print("DATA [INFO] loading {}...".format(Model_name))
        Network = MODELS[Model_name]
        model = Network(weights="imagenet") # load wieghts from imagenet


        for layer in model.layers[:-2]:
            layer.trainable = False

        # Adding custom Layers
        x1 = model.layers[-2].output
        x2 = Dense(1024, activation="relu")(x1)
        x3 = Dense(1024, activation="relu")(x2)
        outputs = Dense(num_classes, activation="sigmoid")(x3)

        # creating the final model
        model_final = Model(input=model.input, output=outputs)
        model_final.summary()
        model_json = model_final.to_json()
        with open("Models\\" + Model_name + "\\model.json", "w") as json_file:
            json_file.write(model_json)


        # compile the model
        model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=Learning_rate, decay=1e-2, momentum=0.9), metrics=["accuracy"])

        # Save the model according to the conditions
        checkpoint_path = output_dir+"\\Models\\"+Model_name+"\\cp-{epoch:04d}."+str(Learning_rate)+".ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=50)
        model_final.save_weights(checkpoint_path.format(epoch=0))
        csv_logger = CSVLogger(output_dir+'\\report\\'+Model_name+'\\log_'+str(Learning_rate)+'.csv', append=False, separator=';')


        fit_model = model_final.fit_generator(augmentaion.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY), steps_per_epoch=len(trainX), epochs=EPOCHS, callbacks = [cp_callback,csv_logger])

        '''
      # plot the training loss and accuracy
        N = np.arange(0, EPOCHS)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["val_loss"], label="val_loss")
        plt.plot(N, H.history["acc"], label="train_acc")
        plt.plot(N, H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy "+ Model_name +"_"+str(Learning_rate))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss\\Accuracy")
        plt.legend()
        plt.savefig(output_dir+'\\Plots\\'+Model_name)
'''

    