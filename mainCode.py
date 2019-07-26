from Plotting import plot
from train import train_model
from test import test_model
import random
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np

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


# Function to convert the the class from binray to decimal
def decimalDecoding(inputdata):
    decoded_data = []
    for i in range(inputdata.shape[0]):
        decoded_data.append(np.argmax(inputdata[i]))
    return np.array(decoded_data)


# train_model()
test_model()
#plot('main')