import pickle
import cv2
import glob
import numpy as np


data_dir = 'Data_set'
dataSet = glob.glob(data_dir+'\\*')
data_Images = []
data_Labels = []

# for loop to loop through the dataset folder
for data in dataSet:
    img = cv2.imread(data)
    image = cv2.resize(img, (224, 224))
    data_Images.append(image.reshape(224, 224, 3))


imageLabel=file.split('\\')[-1].split('_')[-1].split('.')[0]
dataset_Labels.append(imageLabel)

dataX = np.array(data_Images, dtype="float") / 255.0
labels = np.array(dataset_Labels)
print(len(dataset_Labels))

# create a pickle file and write data in it
with open('Nour_pre.pkl', 'wb') as file_pickle:
    pickle.dump([dataX, labels], file_pickle)
file_id.close()


