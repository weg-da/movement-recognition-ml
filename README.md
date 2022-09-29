# Movement-Recognition with ineratial sensor data 
Machine Learning part for hand movement recognition with python
\
\
**Input-Data:** 3-Axis-Accelerometer, 3-Axis-Gyroscope\
**Feature-Extraction:** Time- and Frequency-Domain-Features\
**Classifier:** Support-Vector-Machine  
  
  
## Install requirements with pip
```
pip install -r requirements.txt
```
  
## Running Scripts
### Location of input data
**/input_data/sensor_data.json**

Loading script to extract data to numpy files data.npy and labels.npy
```
/input_data/load_input_data.py 
```
### Feature extraction  

**Time features**  
``` 
feature_extraction_time.py
``` 
**Frequency features**  
run
``` 
feature_extraction_freq.py
```

### Training 
``` 
training.py
``` 
