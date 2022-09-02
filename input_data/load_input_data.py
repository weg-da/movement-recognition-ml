import json
import numpy as np

# load sensor data
with open("sensor_data.json", "r") as f:   
          sensor_data = json.load(f)
          
          
labels = np.zeros(shape= 0)
data = np.zeros((0, 0, 0))
data = []

# loop through sensor data
for list_element in sensor_data:
    labels = np.append(labels, list_element['label'])
    row_element = np.array([np.fromstring(list_element['acc_x'], sep=" "), 
              np.fromstring(list_element['acc_y'], sep=" "),
              np.fromstring(list_element['acc_z'], sep=" "),
              np.fromstring(list_element['gyr_x'], sep=" "),
              np.fromstring(list_element['gyr_y'], sep=" "),
              np.fromstring(list_element['gyr_z'], sep=" "),], dtype = object)
    #data_all = np.append(data_all, row_element)
    data.append(row_element)
    
 
# save data    
np.save('data.npy', data) 
np.save('labels.npy', labels) 


