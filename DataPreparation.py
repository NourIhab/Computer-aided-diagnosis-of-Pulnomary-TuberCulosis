import pickle
import cv2
import glob
import numpy as np

# data preparation(loading the data) for the first time and save it in a pickle file

data_dir = 'Pre_imgs'
data_list = glob.glob(data_dir+'\\*')    # for loading the names to be used in one hot encoding as output layer
data = [] #  array to save the images of the dataset
dataset_Labels = []  # array to save the labels of the dataset images as( normal or abnormal

# for loop to loop through the dataset folder
for file in data_list:
    img = cv2.imread(file)  # read the dataset images file
    image = cv2.resize(img, (224, 224)) ## resize image to 224*224
    data.append(image.reshape(224, 224, 3)) # put all the resized images in a data array

    # first split is to split all the label of the image and the second split is for splitting( 1.png or 0.png) then the third splitting is for( 0 or 1)
    imageLabel=file.split('\\')[-1].split('_')[-1].split('.')[0]  #save the labels that we got in an  array
    dataset_Labels.append(imageLabel)

dataX = np.array(data, dtype="float") / 255.0      # Normalize the data (np.array) convert the data into an array
labels = np.array(dataset_Labels) # convert the labels into an array
print(len(dataset_Labels))

# create a pickle file and open to write in it the data
with open('Nour_pre.pkl', 'wb') as file_id:  # write binary in the pickle file
    pickle.dump([dataX, labels], file_id) # put the normalized data with its labels in the pickle file
file_id.close()  # close the file
#The first one is dump, which dumps an object to a file object and the second one is load, which loads an object from a file object.

