import json
import numpy as np
from scipy import stats
from scipy.fft import fft
from feature_extraction_functions.feature_extraction_funcs import find_main_freq, spectral_entropy
from classifier import predict

import csv
import pickle


with open("sensor_data.json", "r") as f:   #Pickling
          sensor_data = json.load(f)
          
labels = np.zeros(shape= 0)
data = []
    
         
for list_element in sensor_data:
    labels = np.append(labels, list_element['label'])
    print(list_element['label'])
    row_element = np.array([np.fromstring(list_element['acc_x'], sep=" "), 
              np.fromstring(list_element['acc_y'], sep=" "),
              np.fromstring(list_element['acc_z'], sep=" "),
              np.fromstring(list_element['gyr_x'], sep=" "),
              np.fromstring(list_element['gyr_y'], sep=" "),
              np.fromstring(list_element['gyr_z'], sep=" "),], dtype = object)
   
    data.append(row_element)    
    print(predict(row_element[0:3], row_element[3:6]))
    

print(predict(data[0][0:3], data[0][3:6]))
    

    
    

