import numpy as np
from scipy import stats
from feature_extraction_functions.feature_extraction_funcs import zcr

# Loading data
data = np.load('input_data/data.npy', allow_pickle=True)
labels = np.load('input_data/labels.npy', allow_pickle=True) 


# Variables
fs = 200                                    # Sampling rate
ts = 1/fs                                   # Sampling interval
N = 1001                                    # Count of data points in one signal
n_obs = labels.size                         # Total number of observations

t = np.arange(0,(N)/fs, 1/fs)               # time vector
f = np.arange(0, fs/2, fs/N)                # frequency vector   




# Loop to calculate features for each sensor 
# sensor = (1 = Accelerometer, 2 = Gyroscope, 3 = Magnetometer)
for sensor in range (0,6,3):
    x_med = np.zeros(shape= 0)
    y_med = np.zeros(shape= 0)
    z_med = np.zeros(shape= 0)
    
    x_std = np.zeros(shape= 0)
    y_std = np.zeros(shape= 0)
    z_std = np.zeros(shape= 0)
    
    x_var = np.zeros(shape= 0)
    y_var = np.zeros(shape= 0)
    z_var = np.zeros(shape= 0)
    
    x_p25 = np.zeros(shape= 0)
    y_p25 = np.zeros(shape= 0)
    z_p25 = np.zeros(shape= 0)
    
    x_p75 = np.zeros(shape= 0)
    y_p75 = np.zeros(shape= 0)
    z_p75 = np.zeros(shape= 0)
    
    x_min = np.zeros(shape= 0)
    y_min = np.zeros(shape= 0)
    z_min = np.zeros(shape= 0)
    
    x_max = np.zeros(shape= 0)
    y_max = np.zeros(shape= 0)
    z_max = np.zeros(shape= 0)
    
    x_range = np.zeros(shape= 0)
    y_range = np.zeros(shape= 0)
    z_range = np.zeros(shape= 0)
    
    x_range = np.zeros(shape= 0)
    y_range = np.zeros(shape= 0)
    z_range = np.zeros(shape= 0)
    
    x_rms = np.zeros(shape= 0)
    y_rms = np.zeros(shape= 0)
    z_rms = np.zeros(shape= 0)
    
    x_zcr = np.zeros(shape= 0)
    y_zcr = np.zeros(shape= 0)
    z_zcr = np.zeros(shape= 0)
    
    xy_pers = np.zeros(shape= 0)
    xz_pers = np.zeros(shape= 0)
    yz_pers = np.zeros(shape= 0)
    
    xy_xcr_max = np.zeros(shape= 0)
    xz_xcr_max = np.zeros(shape= 0)
    yz_xcr_max = np.zeros(shape= 0)
    
    xy_xcr_std = np.zeros(shape= 0)
    xz_xcr_std = np.zeros(shape= 0)
    yz_xcr_std = np.zeros(shape= 0)
    

# Loop to run through all ovservations
    for i in range(0, n_obs):                # i is the number ob the observation, which is evaluated
       
        x = data[i,sensor]                    # selecting the axes (i): 1-3  Acceleromerter (x,y,z)
        y = data[i,sensor + 1]                #                         4-6  Gyroscope (x,y,z)                    
        z = data[i,sensor + 2]                #                         7-9  Magnetometer (x,y,z)


        ## Standardization
        x = (x - np.min(x))/(np.max(x) - np.min(x))
        y = (y - np.min(y))/(np.max(y) - np.min(y))
        z = (z - np.min(z))/(np.max(z) - np.min(z))


        ## Calculating the features
        # Median
        x_med = np.append(x_med, np.median(x))
        y_med = np.append(y_med, np.median(y))
        z_med = np.append(z_med, np.median(z))
        
        # Standard deviation
        x_std = np.append(x_std, np.std(x, dtype=np.float64))
        y_std = np.append(y_std, np.std(y, dtype=np.float64))
        z_std = np.append(z_std, np.std(z, dtype=np.float64))
        
        # Variance
        x_var = np.append(x_var, np.var(x))
        y_var = np.append(y_var, np.var(y))
        z_var = np.append(z_var, np.var(z))
        
        # Percentile 25th
        x_p25 = np.append(x_p25 , np.percentile(x, 25))
        y_p25 = np.append(y_p25 , np.percentile(y, 25))
        z_p25 = np.append(z_p25 , np.percentile(z, 25))
        
        # Percentile 75th
        x_p75 = np.append(x_p75 , np.percentile(x, 75))
        y_p75 = np.append(y_p75 , np.percentile(y, 75))
        z_p75 = np.append(z_p75 , np.percentile(z, 75))
        
        # Minimum
        x_min = np.append(x_min , np.min(x))
        y_min = np.append(y_min , np.min(y))
        z_min = np.append(z_min , np.min(z))
        
        # Maximum
        x_max = np.append(x_max , np.max(x))
        y_max = np.append(y_max , np.max(y))
        z_max = np.append(z_max , np.max(z))
        
        # Range
        x_range = np.append(x_range , np.max(x) - np.min(x))
        y_range = np.append(y_range , np.max(y) - np.min(y))
        z_range = np.append(z_range , np.max(z) - np.min(z))
        
        # Root mean square
        x_rms = np.append(x_rms, np.sqrt(np.mean(np.square(x))))
        y_rms = np.append(y_rms, np.sqrt(np.mean(np.square(y))))
        z_rms = np.append(z_rms, np.sqrt(np.mean(np.square(z))))
        
        # Zero crossing rate
        x_zcr = np.append(x_zcr , zcr(stats.zscore(x)))
        y_zcr = np.append(y_zcr , zcr(stats.zscore(y)))
        z_zcr = np.append(z_zcr , zcr(stats.zscore(z)))
        
        # Person correlation coefficient
        xy_pers = np.append(xy_pers, np.corrcoef(x,y)[0][1]) 
        xz_pers = np.append(xz_pers, np.corrcoef(x,z)[0][1]) 
        yz_pers = np.append(yz_pers, np.corrcoef(y,z)[0][1]) 
        
        # Cross correlation (maximum)
        xy_xcr_max = np.append(xy_xcr_max, np.correlate(x, y))
        xz_xcr_max = np.append(xz_xcr_max, np.correlate(x, z))
        yz_xcr_max = np.append(yz_xcr_max, np.correlate(y, z))
        
        # Cross correlation (std)
        xy_xcr_std = np.append(xy_xcr_std, np.std(np.correlate(x, y, 'full')))
        xz_xcr_std = np.append(xz_xcr_std, np.std(np.correlate(x, z, 'full')))
        yz_xcr_std = np.append(yz_xcr_std, np.std(np.correlate(y, z, 'full')))
        
    
     
        features = np.array([x_med, y_med, z_med, x_std, y_std, z_std, x_var, y_var, z_var, x_p25, y_p25, z_p25, \
                             x_p75, y_p75, z_p75, x_rms, y_rms, z_rms, x_zcr, y_zcr, z_zcr, xy_pers, xz_pers, yz_pers, \
                             xy_xcr_max, xz_xcr_max, yz_xcr_max, xy_xcr_std, xz_xcr_std, yz_xcr_std ])
    
    if sensor == 0:
        acc_time = features
            
            
    elif sensor == 3:
        gyr_time = features
       
        
time_features = np.concatenate((acc_time, gyr_time), axis = 0)

# Normalization

# save mean and std before normalization for classification of new data
features_mean = np.mean(time_features, axis = 1)
features_std = np.std(time_features, axis = 1)

np.save('extracted_features/features_mean_time.npy', features_mean ) 
np.save('extracted_features/features_std_time.npy', features_std ) 

# normalize
# axis 0 is only one sample is nomalized with itself
# axis 1 features are normalized
time_features_normalized = stats.zscore(time_features, axis=1)


np.save('extracted_features/time_features.npy', time_features_normalized) 
        


        